"""评测器：预测与判分分开。

    # 1. 预测（需要一个 OpenAI 兼容端点，vLLM 起服务即可）
    PYTHONPATH=dataset python3 -m esa.eval predict \
        --endpoint http://localhost:8000/v1 --model <名字> --tag base

    # 2. 判分（纯离线，不需要模型）
    PYTHONPATH=dataset python3 -m esa.eval score --tag base

    # 3. 对比基线与微调后
    PYTHONPATH=dataset python3 -m esa.eval compare --tags base lora

分成两步的理由：推理环境各家不同（vLLM / 本地 transformers / 别的框架），
换环境时只需替换 predict 那一步，判分逻辑不动，也能离线反复调。

**基线必须单独跑一次**：拿未微调的原模型跑 predict --tag base。
LLaMA-Factory 自动画的 loss 曲线不是基线 —— 那只说明模型在拟合训练数据，
不说明工具调用变准了。见交接文档 7.2b。
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from jsonschema import Draft7Validator

from .backend_parser import PARSERS
from .ir import load_schemas, schemas_by_name

EVAL_DIR = Path("dataset/data/eval")


# --------------------------------------------------------------------------
# 预测
# --------------------------------------------------------------------------


def build_messages(rec: dict) -> list[dict]:
    """把评测题渲染成 messages，只喂到模型该出手的地方为止。

    工具调用与工具返回按 **Qwen 模板实际渲染出来的形状**回填（和 render.py 的
    render_wire 一致：调用是 assistant 里的 <tool_call>…</tool_call>，
    返回是 user 里的 <tool_response>…</tool_response>）。

    这么做有两个原因：
    - 保证喂给模型的上下文和训练时见到的一模一样；
    - 避开 OpenAI 协议里 role="tool" 必须带 tool_call_id 的约束，换推理服务不会挂。

    旧写法把 `function_call` 漏进了 `.get(..., "user")` 兜底，助手发出的工具调用会
    被当成用户消息喂进去 —— 以前只有单轮题所以没触发，tool_error 改成喂到失败观测
    之后就会踩到。
    """
    n = rec["gold"]["n_turns_given"]
    msgs = [{"role": "system", "content": rec["system"]}]
    for c in rec["conversations"][:n]:
        tag, value = c["from"], c["value"]
        if tag == "function_call":
            msgs.append({"role": "assistant", "content": f"<tool_call>\n{value}\n</tool_call>"})
        elif tag == "observation":
            msgs.append({"role": "user", "content": f"<tool_response>\n{value}\n</tool_response>"})
        else:
            msgs.append({"role": "assistant" if tag == "gpt" else "user", "content": value})
    return msgs


def call_endpoint(endpoint: str, model: str, messages: list[dict], tools: list,
                  timeout: int = 120) -> str:
    """OpenAI 兼容的 chat/completions。返回原始文本，不让服务端替我们解析工具调用
    —— 评测要考的正是模型自己产出的格式对不对。"""
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "temperature": 0.0,     # 评测必须确定性，否则两次跑分不可比
        "max_tokens": 1024,
    }
    req = urllib.request.Request(
        f"{endpoint.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    msg = data["choices"][0]["message"]
    # 服务端若已把工具调用解析成结构化字段，还原成模型原本的文本形式
    if msg.get("tool_calls"):
        blocks = [
            "<tool_call>\n"
            + json.dumps({"name": tc["function"]["name"],
                          "arguments": json.loads(tc["function"]["arguments"])},
                         ensure_ascii=False)
            + "\n</tool_call>"
            for tc in msg["tool_calls"]
        ]
        return (msg.get("content") or "") + "\n".join(blocks)
    return msg.get("content") or ""


def cmd_predict(args) -> int:
    recs = [json.loads(l) for l in (EVAL_DIR / "eval.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    out_path = EVAL_DIR / f"pred_{args.tag}.jsonl"
    done = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for i, rec in enumerate(recs, 1):
            tools = json.loads(rec["tools"])
            try:
                raw = call_endpoint(args.endpoint, args.model, build_messages(rec), tools)
            except Exception as exc:  # noqa: BLE001
                print(f"  ⚠️  第 {i} 条请求失败：{exc}")
                raw = ""
            fh.write(json.dumps({"id": rec["gold"]["id"], "raw": raw}, ensure_ascii=False) + "\n")
            done += 1
            if done % 20 == 0:
                print(f"  已完成 {done}/{len(recs)}")
    print(f"预测完成 {done} 条 → {out_path}")
    return 0


# --------------------------------------------------------------------------
# 判分
# --------------------------------------------------------------------------


def score(recs: list[dict], preds: dict[str, str], parser_name: str,
          by_name: dict) -> dict:
    parse = PARSERS[parser_name]
    m = Counter()
    confusion: dict[tuple[str, str], int] = defaultdict(int)
    failures: list[dict] = []

    for rec in recs:
        g = rec["gold"]
        raw = preds.get(g["id"], "")
        p = parse(raw)
        got_tools = [c.name for c in p.tool_calls]
        want_tools = g["expected_tools"]
        # 该不该调工具，由标准答案里**还剩几个调用**决定，不看动作字符串。
        # 以前写的是 expected_action == "CALL_TOOL"，于是 RECOVER_TOOL_ERROR 里
        # 「读懂报错、改对参数再调一次」这种正确行为被算成误触发。
        want_call = bool(want_tools)
        is_recover = g["expected_action"] == "RECOVER_TOOL_ERROR"

        m["total"] += 1

        # 1) 格式合法率：解析后至少有工具调用或正文，二者皆空即失败
        ok_format = bool(p.tool_calls or p.content.strip())
        m["format_ok"] += ok_format
        if not ok_format:
            failures.append({"id": g["id"], "why": "解析后为空（格式不合法）", "raw": raw[:200]})

        if want_call:
            m["gold_call"] += 1
            if got_tools:
                # 2) 工具选择准确率
                m["called"] += 1
                right = got_tools[0] == want_tools[0]
                m["tool_correct"] += right
                confusion[(want_tools[0], got_tools[0])] += 1
                if not right:
                    failures.append({"id": g["id"],
                                     "why": f"该调 {want_tools[0]}，实际调了 {got_tools[0]}"})
                # 3) 参数完全匹配 / schema 合法
                if right:
                    m["arg_checked"] += 1
                    m["arg_exact"] += p.tool_calls[0].arguments == g["expected_arguments"][0]
                    spec = by_name.get(got_tools[0])
                    if spec:
                        params = spec["function"].get("parameters", {})
                        errs = list(Draft7Validator(params).iter_errors(p.tool_calls[0].arguments))
                        extra = set(p.tool_calls[0].arguments) - set(params.get("properties", {}))
                        m["arg_schema_ok"] += (not errs and not extra)
            else:
                # 4) 漏调
                m["missed"] += 1
                failures.append({"id": g["id"], "why": f"该调 {want_tools[0]}，但没有调用"})
        elif is_recover:
            # 工具已经失败，正确做法是如实说明 / 换条路，而不是把那个失败调用再发一遍，
            # 也不能编一个结果出来。这类单独统计，**不进 FPR 的分母** ——
            # 误触发率要衡量的是「本来就不该碰工具」，两件事混在一起会让 FPR 不可信。
            m["recover_gold"] += 1
            ok = not got_tools and bool(p.content.strip())
            m["recover_ok"] += ok
            if not ok:
                why = (f"工具已失败，却又调了 {got_tools[0]}" if got_tools
                       else "工具已失败，但没有给出任何说明（正文为空）")
                failures.append({"id": g["id"], "why": why})
        else:
            m["gold_nocall"] += 1
            if got_tools:
                # 5) 误触发 —— 最重要的指标
                m["false_trigger"] += 1
                failures.append({"id": g["id"],
                                 "why": f"不该调用，却调了 {got_tools[0]}（gold={g['expected_action']}）"})
            elif g["expected_action"] == "ASK_USER":
                # 6) 追问命中：没调工具，且确实在提问
                m["ask_gold"] += 1
                m["ask_hit"] += ("？" in p.content or "?" in p.content)

    def rate(a: str, b: str) -> float:
        return round(100.0 * m[a] / m[b], 1) if m[b] else 0.0

    return {
        "样本数": m["total"],
        "格式合法率": rate("format_ok", "total"),
        "工具选择准确率": rate("tool_correct", "called"),
        "误触发率 FPR": rate("false_trigger", "gold_nocall"),
        "漏调率 FNR": rate("missed", "gold_call"),
        "参数完全匹配率": rate("arg_exact", "arg_checked"),
        "参数schema合法率": rate("arg_schema_ok", "arg_checked"),
        "追问命中率": rate("ask_hit", "ask_gold"),
        "工具失败恢复率": rate("recover_ok", "recover_gold"),
        "_confusion": {f"{w}→{gt}": n for (w, gt), n in
                       sorted(confusion.items(), key=lambda x: -x[1]) if w != gt},
        # 两个分母也一并暴露：判分改动最容易出的错就是「某类样本悄悄进错了分母」，
        # test_eval_scoring.py 拿它们做回归断言。
        "_n_nocall": m["gold_nocall"],
        "_n_recover": m["recover_gold"],
        "_failures": failures[:40],
    }


TARGETS = {
    "格式合法率": (100.0, "ge"),
    "工具选择准确率": (90.0, "ge"),
    "误触发率 FPR": (5.0, "le"),
    "漏调率 FNR": (10.0, "le"),
    "参数完全匹配率": (85.0, "ge"),
    "参数schema合法率": (98.0, "ge"),
    "追问命中率": (90.0, "ge"),
    "工具失败恢复率": (90.0, "ge"),
}


def print_report(name: str, r: dict) -> None:
    print(f"\n═══ {name}（{r['样本数']} 条）═══")
    for k, (target, direction) in TARGETS.items():
        v = r[k]
        ok = v >= target if direction == "ge" else v <= target
        sign = "≥" if direction == "ge" else "≤"
        print(f"  {'✅' if ok else '❌'} {k:18s} {v:6.1f}%   目标 {sign}{target}%")
    if r["_confusion"]:
        print("\n  工具混淆（该调→实调，取前 8）：")
        for k, n in list(r["_confusion"].items())[:8]:
            print(f"    {k:52s} {n} 次")


def load_eval() -> list[dict]:
    return [json.loads(l) for l in (EVAL_DIR / "eval.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def load_preds(tag: str) -> dict[str, str]:
    p = EVAL_DIR / f"pred_{tag}.jsonl"
    if not p.exists():
        sys.exit(f"找不到 {p}。先跑：python3 -m esa.eval predict --tag {tag} ...")
    return {json.loads(l)["id"]: json.loads(l)["raw"]
            for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}


def cmd_score(args) -> int:
    schemas, _ = load_schemas(args.schemas)
    r = score(load_eval(), load_preds(args.tag), args.parser, schemas_by_name(schemas))
    print_report(args.tag, r)
    (EVAL_DIR / f"report_{args.tag}.json").write_text(
        json.dumps(r, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if r["_failures"]:
        print(f"\n  失败样例（前 5，全部见 report_{args.tag}.json）：")
        for f in r["_failures"][:5]:
            print(f"    {f['id']}: {f['why']}")
    return 0


def cmd_compare(args) -> int:
    schemas, _ = load_schemas(args.schemas)
    by_name = schemas_by_name(schemas)
    recs = load_eval()
    reports = {t: score(recs, load_preds(t), args.parser, by_name) for t in args.tags}
    for t, r in reports.items():
        print_report(t, r)

    a, b = args.tags[0], args.tags[-1]
    print(f"\n═══ {a} → {b} ═══")
    print(f"  {'指标':20s} {a:>10s} {b:>10s} {'变化':>10s}")
    for k in TARGETS:
        va, vb = reports[a][k], reports[b][k]
        d = vb - va
        better = (d > 0) if TARGETS[k][1] == "ge" else (d < 0)
        mark = "↑" if d > 0 else ("↓" if d < 0 else "—")
        flag = "" if d == 0 else ("  ✅" if better else "  ⚠️")
        print(f"  {k:20s} {va:9.1f}% {vb:9.1f}% {mark}{abs(d):8.1f}{flag}")
    print("\n这张对比表就是《06—效果验证报告》要的「准确性论证」。")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ESA 评测器")
    ap.add_argument("--schemas", default="dataset/schemas/tool_schemas.json")
    ap.add_argument("--parser", default="current", choices=list(PARSERS),
                    help="用哪个后端解析器判分。必须与线上实际使用的一致")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("predict", help="调模型产出预测")
    p.add_argument("--endpoint", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True, help="如 base / lora-1500")
    p.set_defaults(func=cmd_predict)

    s = sub.add_parser("score", help="离线判分")
    s.add_argument("--tag", required=True)
    s.set_defaults(func=cmd_score)

    c = sub.add_parser("compare", help="对比多个 tag")
    c.add_argument("--tags", nargs="+", required=True)
    c.set_defaults(func=cmd_compare)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

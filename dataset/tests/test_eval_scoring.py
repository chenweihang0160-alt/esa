"""判分逻辑的自测：用合成预测验证指标算得对，不需要 GPU。

一个算错分的评测器比没有评测器更糟 —— 它会让你以为模型很好。
所以在拿它去测真模型之前，先用三种已知行为的"假模型"验证：

    perfect   完全照 gold 作答        → 各项应接近满分
    never     从不调用工具            → 漏调率 100%、误触发 0%
    always    见题就调第一个工具      → 误触发率 100%

    python3 dataset/tests/test_eval_scoring.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from esa.eval import EVAL_DIR, score  # noqa: E402
from esa.ir import load_schemas, schemas_by_name  # noqa: E402


def xml_call(name: str, args: dict) -> str:
    """按后端 parse_output 认的 XML 格式渲染一次工具调用。"""
    parts = [f"<tool_call>\n<function={name}>"]
    for k, v in args.items():
        sv = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        parts.append(f"<parameter={k}>\n{sv}\n</parameter>")
    parts.append("</function>\n</tool_call>")
    return "\n".join(parts)


def fake_model(kind: str, rec: dict) -> str:
    g = rec["gold"]
    tools = [t["function"]["name"] for t in json.loads(rec["tools"])]
    if kind == "perfect":
        # 判据是「标准答案里还剩不剩调用」，不是动作字符串 ——
        # RECOVER_TOOL_ERROR 既可能是「改参数重试」（有调用），
        # 也可能是「如实说明」（没有调用）。
        if g["expected_tools"]:
            return xml_call(g["expected_tools"][0], g["expected_arguments"][0])
        if g["expected_action"] == "ASK_USER":
            return "请问是哪一门课程呢？"
        if g["expected_action"] == "RECOVER_TOOL_ERROR":
            return "工具这次没跑通，我不编结果。要不我们换个方式，或者稍后再试一次？"
        return "这个我直接回答就行。"
    if kind == "never":
        return "我直接回答：这个问题的答案是……"
    if kind == "always":
        return xml_call(tools[0], {})
    raise ValueError(kind)


def main() -> int:
    if not (EVAL_DIR / "eval.jsonl").exists():
        print("先跑 PYTHONPATH=dataset python3 -m esa.evalset 生成评测集")
        return 1

    recs = [json.loads(l) for l in (EVAL_DIR / "eval.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    schemas, _ = load_schemas("dataset/schemas/tool_schemas.json")
    by_name = schemas_by_name(schemas)

    results = {}
    for kind in ("perfect", "never", "always"):
        preds = {r["gold"]["id"]: fake_model(kind, r) for r in recs}
        results[kind] = score(recs, preds, "current", by_name)

    p, n, a = results["perfect"], results["never"], results["always"]

    # 分母的期望值直接从评测集本身算出来，不写死数字 —— 数据一变它跟着变。
    want_nocall = sum(1 for r in recs if r["gold"]["expected_action"] in ("ASK_USER", "DIRECT_ANSWER"))
    want_recover = sum(1 for r in recs
                       if r["gold"]["expected_action"] == "RECOVER_TOOL_ERROR"
                       and not r["gold"]["expected_tools"])

    checks = [
        ("perfect: 格式合法率 100%",        p["格式合法率"] == 100.0),
        ("perfect: 工具选择准确率 100%",     p["工具选择准确率"] == 100.0),
        ("perfect: 误触发率 0%",            p["误触发率 FPR"] == 0.0),
        ("perfect: 漏调率 0%",              p["漏调率 FNR"] == 0.0),
        ("perfect: 参数完全匹配率 100%",     p["参数完全匹配率"] == 100.0),
        ("perfect: 参数schema合法率 100%",   p["参数schema合法率"] == 100.0),
        ("perfect: 追问命中率 100%",         p["追问命中率"] == 100.0),
        ("perfect: 无混淆",                 not p["_confusion"]),
        ("never:   漏调率 100%",            n["漏调率 FNR"] == 100.0),
        ("never:   误触发率 0%",            n["误触发率 FPR"] == 0.0),
        ("never:   追问命中率 0%（没问句）",  n["追问命中率"] == 0.0),
        ("always:  误触发率 100%",          a["误触发率 FPR"] == 100.0),
        ("always:  漏调率 0%",              a["漏调率 FNR"] == 0.0),
        ("always:  有混淆记录",             bool(a["_confusion"])),
        ("always:  工具选择准确率 <100%",    a["工具选择准确率"] < 100.0),
        # ---- 以下四项针对「RECOVER_TOOL_ERROR 被判成误触发」那个 bug ----
        # 该报的报：工具已经失败还去调工具，恢复率必须掉到 0
        ("always:  工具失败恢复率 0%",       a["工具失败恢复率"] == 0.0),
        # 不该报的不报：如实说明不重试，恢复率满分
        ("perfect: 工具失败恢复率 100%",     p["工具失败恢复率"] == 100.0),
        # 分母不能串：tool_error 绝不能进 FPR 的分母
        (f"误触发率分母只含追问/直答（{want_nocall}）", p["_n_nocall"] == want_nocall),
        (f"恢复率分母只含无后续调用的 tool_error（{want_recover}）", p["_n_recover"] == want_recover),
    ]

    ok = 0
    for name, passed in checks:
        print(f"{'✅' if passed else '❌'} {name}")
        ok += passed
    print(f"\n{ok} 通过 / {len(checks) - ok} 失败")
    return 0 if ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())

"""数据准入门禁。任何一条数据进训练集之前必须过这里。

设计原则：**失败要吵**。LLaMA-Factory 遇到坏样本只打一行 warning 就静默跳过
（src/llamafactory/data/converter.py:161-181），500 条坏数据就是悄悄少训 500 条，
日志刷过去根本不会有人注意。所以校验必须在数据进仓之前拦下来，并以非零退出码失败。

用法：
    python -m esa.validate dataset/data/ir/*.jsonl
    python -m esa.validate dataset/data/ir/*.jsonl --tokenizer <Qwen3.5权重路径> --cutoff-len 4096
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator

from .ir import (
    CATEGORIES,
    ROLE_ASSISTANT,
    ROLE_TOOL_CALL,
    ROLE_TOOL_RESULT,
    ROLE_USER,
    Sample,
    load_samples,
    load_schemas,
    schemas_by_name,
)
from .render import ROLE_TO_SHAREGPT, to_sharegpt

# ShareGPT 角色顺序的硬性规则（converter.py:144-146）：
# 偶数下标只能是 human/observation，奇数下标只能是 gpt/function_call，且总长为偶。
ODD_TAGS = {"human", "observation"}
EVEN_TAGS = {"gpt", "function_call"}

# math_solver 这些 operation 必须带 variable，但 JSON Schema 表达不了条件必填，
# 只能靠训练数据教会模型 —— 前提是训练数据自己不能错。
MATH_SOLVER_NEEDS_VAR = {"diff", "integrate", "limit", "series", "solve", "summation"}

# 各类别对工具调用次数的约束。类别标错会直接污染评测指标，所以要卡死。
# 这些工具的返回值是"照做的流程"而不是"引用的数据"，grounding 不适用。
GROUNDING_EXEMPT = {"load_skill"}

NO_CALL_CATEGORIES = {"plain_qa", "concept", "hard_negative", "clarify", "refusal"}

PRIVACY_PATTERNS = [
    (re.compile(r"\b1[3-9]\d{9}\b"), "疑似手机号"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "疑似邮箱"),
    (re.compile(r"\b\d{17}[\dXx]\b"), "疑似身份证号"),
]


@dataclass
class Finding:
    sample_id: str
    check: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.check}] {self.sample_id}: {self.detail}"


def _strip_code_blocks(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


def _check_markup(text: str, sid: str, out: list[Finding]) -> None:
    if text.count("```") % 2 != 0:
        out.append(Finding(sid, "code_block", "代码块未闭合（``` 数量为奇数）"))
    body = _strip_code_blocks(text)
    if body.count("$$") % 2 != 0:
        out.append(Finding(sid, "latex", "$$ 定界符不配对"))
    # 去掉 $$ 之后再数单 $
    if body.replace("$$", "").count("$") % 2 != 0:
        out.append(Finding(sid, "latex", "$ 定界符不配对"))


def _check_privacy(text: str, sid: str, out: list[Finding]) -> None:
    for pat, label in PRIVACY_PATTERNS:
        m = pat.search(text)
        if m:
            out.append(Finding(sid, "privacy", f"{label}: {m.group()[:6]}…"))


_NUM = re.compile(r"-?\d+(?:\.\d+)?")
# 时间戳里的数字不是"回答必须引用的值"。不剥掉的话，
# search_core_memories 这种返回体里带 updated_at 的工具会被整批误判 —— 它的有效载荷是文本。
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?")


def _phrases(obj: Any, min_len: int = 4) -> set[str]:
    """抽出结果里够长、够特异的字符串值，作为"回答引用了工具结果"的另一种证据。

    只取字符串**值**，不取字段名 —— 否则 "allowed"、"kp_id" 这类键名会让检查失效。
    """
    out: set[str] = set()

    def walk(o):
        if isinstance(o, str):
            v = o.strip()
            if len(v) >= min_len and not _TIMESTAMP.fullmatch(v):
                out.add(v)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(obj)
    return out


def _numbers(obj: Any) -> set[str]:
    """抽出结果里的数字，用于检查最终回答有没有真的引用工具返回值。"""
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    text = _TIMESTAMP.sub(" ", text)
    out = set()
    for n in _NUM.findall(text):
        out.add(n)
        # 44.0 和 44 视为同一个数
        if n.endswith(".0"):
            out.add(n[:-2])
        elif "." not in n:
            out.add(n + ".0")
    return out


def validate_sample(
    s: Sample,
    by_name: dict[str, dict[str, Any]],
    schema_version: str,
) -> list[Finding]:
    out: list[Finding] = []

    if s.category not in CATEGORIES:
        out.append(Finding(s.id, "category", f"未知类别 {s.category!r}"))
    if not s.template_id:
        out.append(Finding(s.id, "template_id", "缺失 —— 切分时会当成独立组，可能造成泄漏"))
    if s.schema_version != schema_version:
        out.append(
            Finding(s.id, "schema_version", f"样本基于 {s.schema_version}，当前 schema 是 {schema_version}，需重新生成")
        )

    # --- 角色顺序 ---
    tags = [ROLE_TO_SHAREGPT.get(t.role, f"?{t.role}") for t in s.turns]
    if len(tags) % 2 != 0:
        out.append(Finding(s.id, "role_seq", f"消息数为奇数({len(tags)})，LLaMA-Factory 会静默丢弃整条"))
    for i, tag in enumerate(tags):
        allowed = ODD_TAGS if i % 2 == 0 else EVEN_TAGS
        if tag not in allowed:
            out.append(Finding(s.id, "role_seq", f"下标 {i} 是 {tag}，此位置只能是 {sorted(allowed)}"))
            break
    if tags and tags[-1] != "gpt":
        out.append(Finding(s.id, "role_seq", f"最后一条是 {tags[-1]}，必须以 gpt 结尾"))

    # --- 工具调用 ---
    called = s.called_tool_names()
    for name in called:
        if name not in by_name:
            out.append(Finding(s.id, "tool_exists", f"工具 {name!r} 不在当前 schema 里"))
        elif name not in s.tool_names:
            # 极易犯：采样了工具子集，却调用了不在子集里的工具
            out.append(Finding(s.id, "tool_visible", f"调用了 {name!r}，但它不在本样本的 tool_names 里"))

    if s.category in NO_CALL_CATEGORIES and called:
        out.append(Finding(s.id, "category_calls", f"{s.category} 类不应有工具调用，实际调用了 {called}"))
    if s.category == "single_tool_call" and len(called) != 1:
        out.append(Finding(s.id, "category_calls", f"single_tool_call 应恰好 1 次调用，实际 {len(called)} 次"))
    if s.category == "parallel_tool_call":
        widths = [len(t.calls) for t in s.turns if t.role == ROLE_TOOL_CALL]
        if not any(w > 1 for w in widths):
            out.append(Finding(s.id, "category_calls", "parallel_tool_call 需要至少一条消息里有多个并行调用"))

    # --- 参数 schema 校验 ---
    for turn in s.turns:
        for call in turn.calls:
            spec = by_name.get(call.name)
            if not spec:
                continue
            params = spec["function"].get("parameters", {})
            errors = sorted(Draft7Validator(params).iter_errors(call.arguments), key=lambda e: list(e.path))
            for err in errors:
                loc = ".".join(str(p) for p in err.path) or "(root)"
                out.append(Finding(s.id, "arg_schema", f"{call.name}.{loc}: {err.message}"))
            # additionalProperties 默认不禁止，得自己查未知参数
            known = set(params.get("properties", {}))
            for extra in set(call.arguments) - known:
                out.append(Finding(s.id, "arg_unknown", f"{call.name} 出现未知参数 {extra!r}"))
            if call.name == "math_solver":
                op = call.arguments.get("operation")
                if op in MATH_SOLVER_NEEDS_VAR and not call.arguments.get("variable"):
                    out.append(Finding(s.id, "arg_conditional", f"math_solver operation={op} 必须提供 variable"))

    # --- 调用与结果必须配对 ---
    for i, turn in enumerate(s.turns):
        if turn.role != ROLE_TOOL_CALL:
            continue
        nxt = s.turns[i + 1] if i + 1 < len(s.turns) else None
        if nxt is None or nxt.role != ROLE_TOOL_RESULT:
            out.append(Finding(s.id, "call_result", f"下标 {i} 的工具调用后面没有跟 tool_result"))
        elif len(nxt.results) != len(turn.calls):
            out.append(
                Finding(s.id, "call_result", f"下标 {i}: {len(turn.calls)} 个调用但 {len(nxt.results)} 个结果")
            )

    # --- 最终回答必须引用工具返回的真实数值 ---
    final = s.turns[-1].content or "" if s.turns else ""
    last_results = [r for t in s.turns if t.role == ROLE_TOOL_RESULT for r in t.results]
    # 返回值是「要照做的流程」而非「要引用的数据」的工具，不适用 grounding。
    # system prompt 明确要求 load_skill 之后「按正文执行」，复述正文反而是错的。
    last_results = [r for r in last_results if r.name not in GROUNDING_EXEMPT]
    if last_results and not any(r.is_error for r in last_results):
        expected = set().union(*(_numbers(r.content) for r in last_results))
        # 结果为空时（count=0 / 空列表）唯一的数字就是 0，此时"没有查到相关内容"
        # 是完全正确的回答，不该要求它复述一个 0。
        if expected <= {"0", "0.0"}:
            expected = set()
        # 有些工具的有效载荷是文本而非数值（search_core_memories 返回记忆内容、
        # get_weak_prerequisites 返回知识点名）。回答引用了原文同样算 grounded，
        # 只查数字会把这类整批误判。
        phrases = set().union(*(_phrases(r.content) for r in last_results))
        if expected and not (expected & _numbers(final)) and not any(p in final for p in phrases):
            out.append(
                Finding(s.id, "grounding",
                        f"最终回答既没引用工具返回的数值 {sorted(expected)[:3]}，也没引用其返回的文本")
            )

    # --- 文本卫生 ---
    for turn in s.turns:
        if turn.content:
            _check_markup(turn.content, s.id, out)
            if turn.role == ROLE_USER or turn.role == ROLE_ASSISTANT:
                _check_privacy(turn.content, s.id, out)

    return out


# --------------------------------------------------------------------------
# 近重复检测
# --------------------------------------------------------------------------


def simhash(text: str, bits: int = 64) -> int:
    tokens = re.findall(r"[\w一-鿿]+", text.lower())
    grams = [" ".join(tokens[i : i + 2]) for i in range(max(1, len(tokens) - 1))] or tokens
    if not grams:
        return 0
    vec = [0] * bits
    for g in grams:
        h = int.from_bytes(__import__("hashlib").md5(g.encode()).digest()[:8], "big")
        for i in range(bits):
            vec[i] += 1 if (h >> i) & 1 else -1
    return sum(1 << i for i in range(bits) if vec[i] > 0)


def find_near_duplicates(samples: list[Sample], threshold: int = 3) -> list[Finding]:
    """同一 template_id 内部的重复是正常的（就是变体），跨 template_id 的才是问题。"""
    seen: list[tuple[int, str, str]] = []
    out = []
    for s in samples:
        text = " ".join(s.user_queries())
        if not text.strip():
            continue
        h = simhash(text)
        for oh, oid, otpl in seen:
            if s.template_id == otpl:
                continue
            if bin(h ^ oh).count("1") <= threshold:
                out.append(Finding(s.id, "near_dup", f"与 {oid} 高度相似（跨 template_id）"))
                break
        seen.append((h, s.id, s.template_id))
    return out


# --------------------------------------------------------------------------
# 事实核查：概念讲解类数据唯一的自动防线
# --------------------------------------------------------------------------

_BIG_O = re.compile(r"[OΘΩ]\s*\(\s*([^)]{1,24})\s*\)")


def _norm_complexity(expr: str) -> str:
    """把 O(n log n) / O(nlogn) / O(n·log n) / n \\log n 归一成同一个串。

    登记表里写的是完整形式 "O(n log n)"，而正文里抽出来的是括号内的 "n \\log n"，
    所以要先剥掉可能存在的外层 O()/Θ()/Ω() 包装，否则两边永远对不上
    —— 这会让检查看似在工作（什么都报错），实则完全失效。
    """
    s = expr.strip().replace("·", "").replace("*", "").replace("\\", "")
    s = re.sub(r"^[OΘΩoθω]\s*\((.*)\)$", r"\1", s.strip())
    s = re.sub(r"[\s{}$]", "", s.lower())
    return s.replace("^", "").replace("_", "")


def check_verified_facts(samples: list[Sample], facts: dict) -> list[Finding]:
    """核对样本里出现的复杂度断言。

    这是概念讲解数据唯一能自动验的一类事实 —— 生成模型在复杂度上出错是最常见的
    幻觉形式（"快排平均 O(n)"），而结构校验完全抓不到。

    做法：每个知识点在 verified_facts.yaml 里声明它**允许出现**的复杂度集合
    （含最好/平均/最坏/空间）。正文里出现集合之外的复杂度就报错。
    这样既能抓住写错，又不会误伤"顺带提到最坏情况"这种正常表述。
    """
    out = []
    table = {k: {_norm_complexity(c) for c in v} for k, v in facts.get("complexity", {}).items()}
    for s in samples:
        if not s.topic or s.topic not in table:
            continue
        allowed = table[s.topic]
        for turn in s.turns:
            if turn.role != ROLE_ASSISTANT or not turn.content:
                continue
            for raw in _BIG_O.findall(turn.content):
                if _norm_complexity(raw) not in allowed:
                    out.append(
                        Finding(s.id, "fact_complexity",
                                f"知识点 {s.topic} 的正文出现未登记的复杂度 O({raw.strip()})；"
                                f"已登记：{sorted(table[s.topic])}")
                    )
    return out


def _failure_text(content: Any) -> str | None:
    """从一条失败观测里取出报错文案。

    三种失败长相各取各的位置（见 seeds/tool_errors.yaml 文件头）：
      ① 字符串         → 整条就是文案（"[Error]: 搜索请求超时"）
      ② 带 error 键     → error 的值（三个计算器）
      ③ 阻断 / 空结果   → reason 或 note 的值
    """
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        for key in ("error", "reason", "note"):
            if content.get(key):
                return str(content[key])
    return None


def load_error_registry(seeds_path: Path, math_cache_path: Path) -> tuple[set[str], list[re.Pattern]]:
    """报错文案登记表。

    三个来源，都不是人手抄的：
      - seeds/tool_errors.yaml 的 registry 段（逐字取自后端，每条带 src）
      - 同一文件 wrapped 段的 err（自动登记，不用重复写）
      - data/cache/math_real.json 里的 error 值 —— 那是跑后端真实函数抓的，天然合法
    """
    literals: set[str] = set()
    patterns: list[re.Pattern] = []

    if seeds_path.exists():
        import yaml

        cfg = yaml.safe_load(seeds_path.read_text(encoding="utf-8")) or {}
        reg = cfg.get("registry") or {}
        literals |= {str(i["text"]) for i in reg.get("literal") or []}
        patterns += [re.compile(str(i["re"])) for i in reg.get("patterns") or []]
        literals |= {f"[Error]: {i['err']}" for i in cfg.get("wrapped") or []}

    if math_cache_path.exists():
        entries = json.loads(math_cache_path.read_text(encoding="utf-8")).get("entries", {})
        literals |= {v["error"] for v in entries.values()
                     if isinstance(v, dict) and v.get("error")}

    return literals, patterns


def check_error_texts_registered(
    samples: list[Sample], literals: set[str], patterns: list[re.Pattern]
) -> list[Finding]:
    """tool_error 样本里的报错文案必须登记过。

    和 verified_facts 那张复杂度登记表是同一个思路：把「不可自动核验的文本」
    变成可判定的。没有这道检查，任何人都能顺手编一句
    「[Error]: 服务暂时不可用」写进数据 —— 这句话线上永远不会出现，
    模型却会学着去识别它，而真正会出现的那几句反倒没见过。
    """
    out = []
    for s in samples:
        if s.category != "tool_error":
            continue
        for turn in s.turns:
            for r in turn.results:
                if not r.is_error:
                    continue
                text = _failure_text(r.content)
                if text is None:
                    out.append(Finding(s.id, "error_text",
                                       f"{r.name} 标了 is_error，但观测里找不到报错文案"))
                    continue
                if text in literals or any(p.search(text) for p in patterns):
                    continue
                out.append(Finding(
                    s.id, "error_text",
                    f"{r.name} 的报错文案未登记：{text[:60]!r}。"
                    f"先在 seeds/tool_errors.yaml 的 registry 段登记（带 src 行号），不要凭印象编报错"))
    return out


def check_exact_duplicates(samples: list[Sample]) -> list[Finding]:
    """字面完全相同的用户话术。

    近重复检测（simhash）只比较**不同** template_id 之间，同一模板内的字面重复
    抓不到 —— 曾经因此在 680 条里混进 171 条完全一样的样本而无人察觉。
    """
    seen: dict[str, str] = {}
    out = []
    for s in samples:
        key = " | ".join(s.user_queries()).strip()
        if not key:
            continue
        if key in seen:
            out.append(Finding(s.id, "exact_dup", f"用户话术与 {seen[key]} 完全相同：{key[:40]}…"))
        else:
            seen[key] = s.id
    return out


def check_revision_uses_latest(samples: list[Sample]) -> list[Finding]:
    """「用户修改参数」类样本必须用最新值调用工具。

    这类 State 存在的意义就是教「用最新值」，一旦用了被覆盖的旧值就是教反了。
    做法：最后一轮用户话术里出现的取值，必须能在工具参数里找到。
    """
    out = []
    for s in samples:
        if "修改参数" not in s.template_id:
            continue
        users = [t.content or "" for t in s.turns if t.role == ROLE_USER]
        calls = [c for t in s.turns for c in t.calls]
        if len(users) < 2 or not calls:
            continue
        latest, args = users[-1], calls[0].arguments
        vals = {str(v) for v in args.values()}
        # 最后一轮里提到的、且确实出现在参数域里的值，必须是被采用的那个
        if not any(v in latest for v in vals):
            out.append(
                Finding(s.id, "revision_stale",
                        f"最后一轮用户话术 {latest[:30]!r} 里的新值没有出现在工具参数 {args} 中，疑似用了旧值")
            )
    return out


# --------------------------------------------------------------------------
# 长度检查
# --------------------------------------------------------------------------


def check_lengths(samples, by_name, cutoff_len: int, tokenizer_path: str | None) -> list[Finding]:
    """长度必须按 token 数算，不是字符数。

    没装 transformers 时退化成字符粗估，并明确告知这只是估算 —— 真正的判定
    必须在有权重的机器上用真 tokenizer 跑一遍（P0-3）。
    """
    out = []
    tok = None
    if tokenizer_path:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        except Exception as exc:  # noqa: BLE001
            out.append(Finding("*", "tokenizer", f"加载失败，退回字符估算: {exc}"))

    for s in samples:
        rec = to_sharegpt(s, by_name)
        blob = rec["system"] + rec["tools"] + "".join(c["value"] for c in rec["conversations"])
        if tok is not None:
            n = len(tok(blob, add_special_tokens=False)["input_ids"])
            label = "token"
        else:
            cjk = len(re.findall(r"[一-鿿]", blob))
            n = int(cjk + (len(blob) - cjk) / 3.5)
            label = "token(估算)"
        if n > cutoff_len:
            out.append(Finding(s.id, "length", f"{label} {n} > cutoff_len {cutoff_len}，会被截断"))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ESA 数据集校验器")
    ap.add_argument("files", nargs="+", help="IR jsonl 文件")
    ap.add_argument("--schemas", default="dataset/schemas/tool_schemas.json")
    ap.add_argument("--facts", default="dataset/seeds/verified_facts.yaml")
    ap.add_argument("--error-registry", default="dataset/seeds/tool_errors.yaml")
    ap.add_argument("--math-cache", default="dataset/data/cache/math_real.json")
    # 8192 而不是 4096：真实 get_mastery_report 一次返回 weak/strong/stale 各 5 条完整
    # _row_state（16 字段），约 1851 tokens；再叠上 22 个工具的 schema 区，4096 装不下。
    # 注意这是"迁就现状"，不是最优解 —— 见 README 里给后端的裁剪建议。
    ap.add_argument("--cutoff-len", type=int, default=8192)
    ap.add_argument("--tokenizer", default=None, help="Qwen3.5 权重路径，给了就用真 tokenizer 数 token")
    ap.add_argument("--no-dup-check", action="store_true")
    args = ap.parse_args(argv)

    schemas, version = load_schemas(args.schemas)
    by_name = schemas_by_name(schemas)

    samples: list[Sample] = []
    findings: list[Finding] = []
    for f in args.files:
        try:
            samples.extend(load_samples(f))
        except ValueError as exc:
            findings.append(Finding(f, "parse", str(exc)))

    ids = Counter(s.id for s in samples)
    findings += [Finding(i, "dup_id", f"id 重复 {n} 次") for i, n in ids.items() if n > 1]

    for s in samples:
        findings += validate_sample(s, by_name, version)
    findings += check_exact_duplicates(samples)
    findings += check_revision_uses_latest(samples)

    registry_path = Path(args.error_registry)
    literals, patterns = load_error_registry(registry_path, Path(args.math_cache))
    if literals or patterns:
        findings += check_error_texts_registered(samples, literals, patterns)
    elif any(s.category == "tool_error" for s in samples):
        findings.append(Finding("*", "error_text",
                                f"有 tool_error 样本但找不到登记表 {registry_path}，报错文案无人核对"))
    if not args.no_dup_check:
        findings += find_near_duplicates(samples)
    findings += check_lengths(samples, by_name, args.cutoff_len, args.tokenizer)

    facts_path = Path(args.facts)
    if facts_path.exists():
        import yaml

        findings += check_verified_facts(samples, yaml.safe_load(facts_path.read_text(encoding="utf-8")))
    else:
        findings.append(Finding("*", "facts", f"找不到 {facts_path}，跳过事实核查"))

    print(f"schema_version = {version}")
    print(f"样本数 = {len(samples)}")
    by_cat = Counter(s.category for s in samples)
    for cat, n in sorted(by_cat.items()):
        print(f"  {cat:20s} {n}")
    if (n_review := sum(1 for s in samples if s.needs_review)) :
        print(f"\n待人工复核 {n_review} 条（无法自动核验的写作类内容）")

    if not findings:
        print("\n✅ 全部通过")
        return 0

    grouped = Counter(f.check for f in findings)
    print(f"\n❌ {len(findings)} 处问题：")
    for check, n in grouped.most_common():
        print(f"  {check:18s} {n}")
    print()
    for f in findings[:60]:
        print(" ", f)
    if len(findings) > 60:
        print(f"  … 另有 {len(findings) - 60} 处")
    return 1


if __name__ == "__main__":
    sys.exit(main())

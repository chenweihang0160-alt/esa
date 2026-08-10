"""后端 parser 兼容性验证 —— 不需要 GPU、不需要下权重就能跑。

结论先说：会议事程.md 里的 parse_output 无法解析 LLaMA-Factory qwen 模板训练出的输出，
而且失败方式是**静默返回空对象**，没有任何异常。

这个脚本把两种线上格式分别喂给现有 parser 和修复版 parser，把差异打出来。
拿它去和后端对齐 P0-1，比口头描述有效得多。

    python dataset/tests/test_parser_compat.py
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from esa.ir import load_samples, load_schemas, schemas_by_name  # noqa: E402
from esa.render import assistant_wire_segments  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# 后端现有实现（原样抄自 会议事程.md:104-138，只把 models 换成本地 dataclass）
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    name: str
    arguments: dict


@dataclass
class ParsedOutput:
    reasoning: str = ""
    content: str = ""
    tool_calls: list = field(default_factory=list)


def _try_cast(value: str):
    value = value.strip()
    if not value:
        return ""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def parse_output_current(raw_text: str) -> ParsedOutput:
    """后端当前实现。期望 <tool_call><function=X><parameter=K>v</parameter></function></tool_call>"""
    result = ParsedOutput()

    think_match = re.search(r"(?:<think>)?(.*?)</think>", raw_text, re.DOTALL)
    if think_match:
        result.reasoning = think_match.group(1).strip()

    tool_call_blocks = re.findall(r"<tool_call>(.*?)</tool_call>", raw_text, re.DOTALL)

    if not tool_call_blocks:
        remaining = re.sub(r"(?:<think>)?.*?</think>", "", raw_text, flags=re.DOTALL)
        result.content = remaining.strip() or raw_text.strip()
        return result

    for block in tool_call_blocks:
        func_match = re.search(r"<function=([^>\s]+)>", block)
        if not func_match:
            continue  # ← 问题就在这里：匹配不到就跳过，最后返回一个完全空的对象
        func_name = func_match.group(1)
        param_matches = re.findall(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", block, re.DOTALL)
        args = {k: _try_cast(v) for k, v in param_matches}
        result.tool_calls.append(ToolCall(name=func_name, arguments=args))

    return result


# ---------------------------------------------------------------------------
# 建议的修复版：解析 Qwen 原生 JSON，同时向后兼容 XML
# ---------------------------------------------------------------------------


def parse_output_fixed(raw_text: str) -> ParsedOutput:
    """兼容两种格式，且解析失败时不静默吞掉内容。"""
    result = ParsedOutput()

    think_match = re.search(r"(?:<think>)?(.*?)</think>", raw_text, re.DOTALL)
    if think_match:
        result.reasoning = think_match.group(1).strip()
    body = re.sub(r"(?:<think>)?.*?</think>", "", raw_text, flags=re.DOTALL)

    blocks = re.findall(r"<tool_call>(.*?)</tool_call>", body, re.DOTALL)
    if not blocks:
        result.content = body.strip()
        return result

    for block in blocks:
        block = block.strip()
        # 优先按 Qwen 原生 JSON 解析
        try:
            payload = json.loads(block)
            calls = payload if isinstance(payload, list) else [payload]
            for c in calls:
                if "name" in c:
                    result.tool_calls.append(ToolCall(name=c["name"], arguments=c.get("arguments", {})))
            continue
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
        # 回退到 XML 风格
        func_match = re.search(r"<function=([^>\s]+)>", block)
        if func_match:
            params = re.findall(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", block, re.DOTALL)
            result.tool_calls.append(
                ToolCall(name=func_match.group(1), arguments={k: _try_cast(v) for k, v in params})
            )

    # 关键：解析不出任何工具调用时，不能让 content 也是空的
    if not result.tool_calls:
        result.content = body.strip()

    return result


# ---------------------------------------------------------------------------


def main() -> int:
    schemas, _ = load_schemas(ROOT / "dataset/schemas/tool_schemas.json")
    by_name = schemas_by_name(schemas)
    samples = load_samples(ROOT / "dataset/data/ir/calculators.jsonl")
    sample = next(s for s in samples if s.category == "single_tool_call")

    expected_name = sample.called_tool_names()[0]
    expected_args = sample.turns[1].calls[0].arguments

    print("=" * 74)
    print(f"样本 {sample.id}   期望解析出: {expected_name}({expected_args})")
    print("=" * 74)

    failures = 0
    for fmt in ("qwen", "xml"):
        # 第一段助手输出就是工具调用，这是要喂给 parse_output 的关键那一段
        wire = assistant_wire_segments(sample, by_name, tool_format=fmt)[0]
        print(f"\n模型实际输出（tool_format={fmt}）:")
        print("  " + wire.strip().replace("\n", "\n  "))

        for label, fn in (("当前 parse_output", parse_output_current), ("修复版 parse_output", parse_output_fixed)):
            got = fn(wire)
            ok = len(got.tool_calls) == 1 and got.tool_calls[0].name == expected_name
            mark = "✅" if ok else "❌"
            detail = (
                f"{got.tool_calls[0].name}({got.tool_calls[0].arguments})"
                if got.tool_calls
                else f"tool_calls=[] content={got.content[:30]!r}"
            )
            print(f"  {mark} {label:20s} → {detail}")
            if not ok and label.startswith("修复版"):
                failures += 1
            if not ok and fmt == "qwen" and label.startswith("当前"):
                print("     ↑ 这就是问题：无异常、无日志，前端拿到完全空白的回复")

    print("\n" + "=" * 74)
    print("结论：LLaMA-Factory 的 qwen 系模板产出 JSON 格式（data/tool_utils.py:423-428）。")
    print("      当前 parser 只认 XML，遇到 JSON 会静默返回空 ParsedOutput。")
    print("      建议方案 A：改 parse_output 为上面的修复版（同时兼容两种格式），")
    print("      训练数据侧零改动，且与 vLLM --tool-call-parser hermes 天然一致。")
    print("=" * 74)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

"""后端 parse_output 的两个版本，供评测和兼容性测试共用。

`parse_output_current` 逐字复刻 backend/core/utils/parser.py（2026-08-10 快照）。
`parse_output_dual` 是建议的修复版：同时认 Qwen 原生 JSON 和现有 XML。

为什么要有两份：LLaMA-Factory 的 qwen 模板产出 JSON，后端只认 XML，
两者不兼容且失败时静默返回空对象。评测必须用**后端实际会用的那个解析器**，
否则测出来的分数和线上表现对不上。见交接文档 6.1。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


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
    """复刻后端 _try_cast：含 DeepSeek 系 Python 字面量的归一化。"""
    value = value.strip()
    if not value:
        return ""
    aliases = {"true": True, "false": False, "none": None, "null": None}
    if value.casefold() in aliases:
        return aliases[value.casefold()]
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def parse_output_current(raw_text: str) -> ParsedOutput:
    """后端当前实现。只认 <function=X><parameter=K>v</parameter> 这种 XML。

    ⚠️ 匹配不到时 `continue`，最终 tool_calls 为空且 content 也从未赋值 ——
    返回一个完全空的对象，无异常无日志。
    """
    result = ParsedOutput()
    m = re.search(r"(?:<think>)?(.*?)</think>", raw_text, re.DOTALL)
    if m:
        result.reasoning = m.group(1).strip()

    blocks = re.findall(r"<tool_call>(.*?)</tool_call>", raw_text, re.DOTALL)
    if not blocks:
        remaining = re.sub(r"(?:<think>)?.*?</think>", "", raw_text, flags=re.DOTALL)
        result.content = remaining.strip() or raw_text.strip()
        return result

    for block in blocks:
        fm = re.search(r"<function=([^>\s]+)>", block)
        if not fm:
            continue
        params = re.findall(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", block, re.DOTALL)
        result.tool_calls.append(
            ToolCall(name=fm.group(1), arguments={k: _try_cast(v) for k, v in params})
        )
    return result


def parse_output_dual(raw_text: str) -> ParsedOutput:
    """建议的修复版：先试 Qwen 原生 JSON，再回退 XML；且解析不出调用时不吞掉正文。"""
    result = ParsedOutput()
    m = re.search(r"(?:<think>)?(.*?)</think>", raw_text, re.DOTALL)
    if m:
        result.reasoning = m.group(1).strip()
    body = re.sub(r"(?:<think>)?.*?</think>", "", raw_text, flags=re.DOTALL)

    blocks = re.findall(r"<tool_call>(.*?)</tool_call>", body, re.DOTALL)
    if not blocks:
        result.content = body.strip()
        return result

    for block in blocks:
        block = block.strip()
        try:
            payload = json.loads(block)
            for c in payload if isinstance(payload, list) else [payload]:
                if isinstance(c, dict) and "name" in c:
                    result.tool_calls.append(
                        ToolCall(name=c["name"], arguments=c.get("arguments", {}))
                    )
            continue
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
        fm = re.search(r"<function=([^>\s]+)>", block)
        if fm:
            params = re.findall(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", block, re.DOTALL)
            result.tool_calls.append(
                ToolCall(name=fm.group(1), arguments={k: _try_cast(v) for k, v in params})
            )

    if not result.tool_calls:
        result.content = body.strip()
    return result


PARSERS = {"current": parse_output_current, "dual": parse_output_dual}

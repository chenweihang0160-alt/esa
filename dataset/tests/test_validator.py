"""校验器的负向测试：故意造坏数据，确认每种错误都能被抓到。

一个从不报错的校验器等于没有校验器。每加一条新检查，都要在这里补一条对应的坏样本。

    python dataset/tests/test_validator.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from esa.ir import Sample, ToolCall, ToolResult, Turn, load_schemas, schemas_by_name  # noqa: E402
from esa.validate import validate_sample  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCHEMAS, VERSION = load_schemas(ROOT / "dataset/schemas/tool_schemas.json")
BY_NAME = schemas_by_name(SCHEMAS)


def make(**over) -> Sample:
    """一条本身完全合法的样本，测试时逐项破坏它。"""
    base = dict(
        id="t1",
        template_id="tpl",
        category="single_tool_call",
        schema_version=VERSION,
        system="你是 ESA。",
        tool_names=["calculator", "web_search"],
        turns=[
            Turn(role="user", content="算一下 2+2"),
            Turn(role="tool_call", calls=[ToolCall("calculator", {"expression": "2+2"})]),
            Turn(role="tool_result", results=[ToolResult("calculator", {"expression": "2+2", "result": 4})]),
            Turn(role="assistant", content="结果是 $4$。"),
        ],
    )
    base.update(over)
    return Sample(**base)


CASES: list[tuple[str, Sample, str]] = [
    (
        "基准样本应当通过",
        make(),
        "",
    ),
    (
        "消息数为奇数",
        make(turns=make().turns[:3]),
        "role_seq",
    ),
    (
        "工具调用后面没跟 tool_result",
        make(turns=[Turn(role="user", content="算 2+2"), Turn(role="tool_call", calls=[ToolCall("calculator", {"expression": "2+2"})])]),
        "call_result",
    ),
    (
        "调用了 schema 里不存在的工具",
        make(
            tool_names=["nonexistent_tool"],
            turns=[
                Turn(role="user", content="x"),
                Turn(role="tool_call", calls=[ToolCall("nonexistent_tool", {})]),
                Turn(role="tool_result", results=[ToolResult("nonexistent_tool", {})]),
                Turn(role="assistant", content="好的。"),
            ],
        ),
        "tool_exists",
    ),
    (
        "调用了不在本样本 tool_names 里的工具",
        make(tool_names=["web_search"]),
        "tool_visible",
    ),
    (
        "缺必填参数",
        make(
            turns=[
                Turn(role="user", content="算一下"),
                Turn(role="tool_call", calls=[ToolCall("calculator", {})]),
                Turn(role="tool_result", results=[ToolResult("calculator", {"expression": "2+2", "result": 4})]),
                Turn(role="assistant", content="结果是 $4$。"),
            ]
        ),
        "arg_schema",
    ),
    (
        "出现未知参数",
        make(
            turns=[
                Turn(role="user", content="算一下 2+2"),
                Turn(role="tool_call", calls=[ToolCall("calculator", {"expression": "2+2", "precision": 3})]),
                Turn(role="tool_result", results=[ToolResult("calculator", {"expression": "2+2", "result": 4})]),
                Turn(role="assistant", content="结果是 $4$。"),
            ]
        ),
        "arg_unknown",
    ),
    (
        "math_solver 条件必填缺 variable",
        make(
            tool_names=["math_solver"],
            turns=[
                Turn(role="user", content="求导"),
                Turn(role="tool_call", calls=[ToolCall("math_solver", {"operation": "diff", "expression": "x**2"})]),
                Turn(role="tool_result", results=[ToolResult("math_solver", {"result": "2*x"})]),
                Turn(role="assistant", content="导数是 $2x$。"),
            ],
        ),
        "arg_conditional",
    ),
    (
        "hard_negative 类别却调了工具",
        make(category="hard_negative"),
        "category_calls",
    ),
    (
        "最终回答没引用工具返回值（编造）",
        make(
            turns=[
                Turn(role="user", content="算一下 2+2"),
                Turn(role="tool_call", calls=[ToolCall("calculator", {"expression": "2+2"})]),
                Turn(role="tool_result", results=[ToolResult("calculator", {"expression": "2+2", "result": 4})]),
                Turn(role="assistant", content="结果是 $99$。"),
            ]
        ),
        "grounding",
    ),
    (
        "LaTeX 定界符不配对",
        make(
            turns=[
                Turn(role="user", content="算一下 2+2"),
                Turn(role="tool_call", calls=[ToolCall("calculator", {"expression": "2+2"})]),
                Turn(role="tool_result", results=[ToolResult("calculator", {"expression": "2+2", "result": 4})]),
                Turn(role="assistant", content="结果是 $4。"),
            ]
        ),
        "latex",
    ),
    (
        "schema 版本对不上",
        make(schema_version="deadbeef"),
        "schema_version",
    ),
    (
        "回答里混入手机号",
        make(
            turns=[
                Turn(role="user", content="算一下 2+2"),
                Turn(role="tool_call", calls=[ToolCall("calculator", {"expression": "2+2"})]),
                Turn(role="tool_result", results=[ToolResult("calculator", {"expression": "2+2", "result": 4})]),
                Turn(role="assistant", content="结果是 $4$，有问题打 13812345678。"),
            ]
        ),
        "privacy",
    ),
]


def corpus_cases() -> list[tuple[str, bool]]:
    """语料级检查（跨样本），单条 validate_sample 查不到这些。"""
    from esa.validate import (
        check_error_texts_registered,
        check_exact_duplicates,
        check_revision_uses_latest,
        check_verified_facts,
    )

    results = []

    # 1) 字面完全相同的两条样本必须被抓到
    a, b = make(id="d1"), make(id="d2")
    results.append(("字面重复的用户话术", bool(check_exact_duplicates([a, b]))))
    results.append(("话术不同则不应报重复",
                    not check_exact_duplicates([a, make(id="d3", turns=[
                        Turn(role="user", content="换个问法算 2+2"),
                        *make().turns[1:]])])))

    # 2) 「修改参数」用了旧值必须被抓到
    stale = Sample(
        id="r1", template_id="S002__修改参数__x", category="multi_turn_tool",
        schema_version=VERSION, system="你是 ESA。", tool_names=["recommend_practice"],
        turns=[
            Turn(role="user", content="我想复习数据结构，还有 8 周考试。"),
            Turn(role="assistant", content="好的。"),
            Turn(role="user", content="改一下，我要问的是操作系统"),
            # ↓ 仍然用旧课程，属于教反了
            Turn(role="tool_call", calls=[ToolCall("recommend_practice", {"course": "数据结构", "weeks_to_exam": 8})]),
            Turn(role="tool_result", results=[ToolResult("recommend_practice", {"recommendations": []})]),
            Turn(role="assistant", content="好的，已为你安排。"),
        ],
    )
    results.append(("修改参数却用了旧值", bool(check_revision_uses_latest([stale]))))

    fixed = Sample(**{**stale.__dict__, "id": "r2"})
    fixed.turns = list(stale.turns)
    fixed.turns[3] = Turn(role="tool_call",
                          calls=[ToolCall("recommend_practice", {"course": "操作系统", "weeks_to_exam": 8})])
    results.append(("修改参数用了新值则通过", not check_revision_uses_latest([fixed])))

    # 3) 复杂度断言核查
    facts = {"complexity": {"quicksort_analysis": ["O(n log n)", "O(n^2)", "O(log n)"]}}
    wrong = make(id="f1", topic="quicksort_analysis", turns=[
        Turn(role="user", content="快排平均复杂度"),
        Turn(role="assistant", content="快速排序平均时间复杂度是 $O(n)$。"),
    ])
    right = make(id="f2", topic="quicksort_analysis", turns=[
        Turn(role="user", content="快排平均复杂度"),
        Turn(role="assistant", content="平均 $O(n \\log n)$，最坏 $O(n^2)$。"),
    ])
    results.append(("复杂度写错（O(n)）", bool(check_verified_facts([wrong], facts))))
    results.append(("复杂度正确且含对比项", not check_verified_facts([right], facts)))

    # 4) 报错文案登记核查
    #
    # 正反两个用例缺一不可：只写「该报错的会报」会得到一个永远报错的假检查
    # （5.9 那次就是这么栽的：登记表带 O(...) 包装、正文抽出来不带，两边永远对不上，
    #  看起来一直在工作，实则完全失效）。
    literals = {"[Error]: 搜索请求超时", "当前会话为 isolated 模式，禁止读取长期记忆"}
    patterns = [re.compile(r"^未找到课程 '.+' 的知识点，请确认课程名$")]

    def toolerr(sid: str, content) -> Sample:
        return make(id=sid, category="tool_error", tool_names=["web_search"], turns=[
            Turn(role="user", content="搜一下最新的大模型进展"),
            Turn(role="tool_call", calls=[ToolCall("web_search", {"query": "大模型 最新进展"})]),
            Turn(role="tool_result", results=[ToolResult("web_search", content, is_error=True)]),
            Turn(role="assistant", content="搜索没跑通，我不会用猜测代替检索结果。"),
        ])

    made_up = toolerr("e1", "[Error]: 服务暂时不可用，请稍后重试")   # 线上不存在的句子
    registered = toolerr("e2", "[Error]: 搜索请求超时")               # web_search.py:94 的原文
    by_pattern = toolerr("e3", {"count": 0, "recommendations": [],
                                "note": "未找到课程 '高等数学' 的知识点，请确认课程名"})
    no_text = toolerr("e4", {"count": 0, "recommendations": []})      # 标了 is_error 却没有文案

    results.append(("编造的报错文案被拦下",
                    bool(check_error_texts_registered([made_up], literals, patterns))))
    results.append(("登记过的报错文案放行",
                    not check_error_texts_registered([registered], literals, patterns)))
    results.append(("正则登记的报错文案放行",
                    not check_error_texts_registered([by_pattern], literals, patterns)))
    results.append(("标了 is_error 却没有报错文案被拦下",
                    bool(check_error_texts_registered([no_text], literals, patterns))))
    results.append(("非 tool_error 类不受此检查影响",
                    not check_error_texts_registered([make(id="e5")], literals, patterns)))
    return results


def main() -> int:
    passed = failed = 0
    for name, sample, expect in CASES:
        checks = {f.check for f in validate_sample(sample, BY_NAME, VERSION)}
        ok = (not checks) if expect == "" else (expect in checks)
        print(f"{'✅' if ok else '❌'} {name:36s} 命中={sorted(checks) or '无'}")
        if ok:
            passed += 1
        else:
            failed += 1
            print(f"     期望命中 {expect!r}")

    print()
    for name, ok in corpus_cases():
        print(f"{'✅' if ok else '❌'} {name}")
        passed, failed = (passed + 1, failed) if ok else (passed, failed + 1)

    print(f"\n{passed} 通过 / {failed} 失败")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

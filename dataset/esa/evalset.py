"""评测集构建：按行为类别显式挑选，不靠随机切分。

为什么不用 split.py 的随机分层：
  tool_error 只有 8 条、clarify 的 template 组也少，随机切分下它们要么整组进 train、
  要么在 val/test 里一条不剩 —— 前面已经反复出现"test 里没有 clarify 类样本"的告警。
  评测集本来就该按"要考什么行为"来挑，而不是碰运气。

三层设计（对应交接文档 7.x）：
  L1 同分布   每个 State 留出部分事实组合         → 规律学会了没有
  L2 状态外推 整组留出某些 template               → 规律能否泛化到没见过的组合
  L3 场景外推 留出整个场景/工具                   → 能否泛化到新任务

**评测集绝不能进训练集**：build() 会同时产出 eval 和 train，并断言两者的
template_id 不相交。这是防泄题的唯一保障。

用法：
    PYTHONPATH=dataset python3 -m esa.evalset --out dataset/data/eval
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from .ir import (
    ROLE_ASSISTANT,
    ROLE_TOOL_CALL,
    ROLE_TOOL_RESULT,
    Sample,
    iter_ir_files,
    load_samples,
    load_schemas,
    schemas_by_name,
)
from .render import to_sharegpt

# 每个类别至少要有多少条进评测集。低于这个数，该行为的指标就没有统计意义。
QUOTA = {
    "single_tool_call": 90,
    "clarify": 30,
    "hard_negative": 40,
    "multi_turn_tool": 20,
    "tool_error": 12,
}

# L3 场景外推：这些 template 前缀整体留出，训练时完全不给。
# 选 get_review_timing 是因为它有独立的正例组，且与 get_mastery_level 构成混淆对
# —— 正好能测"模型有没有真正理解两者区别"，而不是背下了见过的问法。
L3_HOLDOUT_PREFIXES = ("get_review_timing__",)


def expected_action(s: Sample) -> str:
    """从样本反推标准动作，作为评测的 gold label。

    与设计总表 00 页的决策树同一套词表。
    """
    if not s.called_tool_names():
        return "ASK_USER" if s.category == "clarify" else "DIRECT_ANSWER"
    if s.category == "tool_error":
        return "RECOVER_TOOL_ERROR"
    return "CALL_TOOL"


def cut_point(s: Sample) -> int:
    """评测题喂到第几条消息为止（不含这一条）。

    一般样本切在「模型第一次该出手」的位置。

    **tool_error 是例外，而且此前切错了。** 这类 State 的前提是工具已经执行并失败，
    要考的是「拿到失败观测之后怎么办」。如果也切在用户话之后，那一刻的正确行为其实是
    照常调用工具（模型还不知道它会失败），而 gold 写着 RECOVER_TOOL_ERROR ——
    判分会把正确行为算成误触发，污染 FPR 这个最重要的指标。
    所以对 tool_error 切到**第一条失败观测之后**。
    """
    first_move = next(
        (i for i, t in enumerate(s.turns) if t.role in (ROLE_TOOL_CALL, ROLE_ASSISTANT)),
        len(s.turns),
    )
    if s.category != "tool_error":
        return first_move
    for i, t in enumerate(s.turns):
        if t.role == ROLE_TOOL_RESULT and any(r.is_error for r in t.results):
            return i + 1
    return first_move


def gold_of(s: Sample) -> dict:
    """一条评测题的标准答案。"""
    n = cut_point(s)
    # 标准答案只包含切点**之后**该发生的调用。
    # 「如实说明不重试」的样本这里是空的；「改参数重试」的样本是那一次修正后的调用。
    calls = [c for t in s.turns[n:] for c in t.calls]
    return {
        "id": s.id,
        "category": s.category,
        "template_id": s.template_id,
        "expected_action": expected_action(s),
        "expected_tools": [c.name for c in calls],
        "expected_arguments": [c.arguments for c in calls],
        "n_turns_given": n,
    }


def build(
    samples: list[Sample],
    seed: int = 20260810,
) -> tuple[list[Sample], list[Sample], dict]:
    """返回 (评测集, 训练集, 统计)。两者的 template_id 保证不相交。"""
    rng = random.Random(seed)

    by_tpl: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_tpl[s.template_id].append(s)

    eval_tpls: set[str] = set()
    layer: dict[str, str] = {}

    # ---- L3：整组留出 ----
    for tpl in by_tpl:
        if tpl.startswith(L3_HOLDOUT_PREFIXES):
            eval_tpls.add(tpl)
            layer[tpl] = "L3_场景外推"

    # ---- L1 / L2：按类别配额挑 template ----
    tpls_by_cat: dict[str, list[str]] = defaultdict(list)
    for tpl, items in by_tpl.items():
        if tpl not in eval_tpls:
            tpls_by_cat[items[0].category].append(tpl)

    for cat, want in QUOTA.items():
        pool = sorted(tpls_by_cat.get(cat, []))
        rng.shuffle(pool)
        taken = 0
        for i, tpl in enumerate(pool):
            if taken >= want:
                break
            # 该类别只剩最后一个 template 时不能全拿走，否则训练集里这个行为归零
            if len(pool) - i <= 1:
                break
            eval_tpls.add(tpl)
            # 大组（同一事实组合的多种问法）算 L1，小组算 L2
            layer[tpl] = "L1_同分布" if len(by_tpl[tpl]) >= 3 else "L2_状态外推"
            taken += len(by_tpl[tpl])

    eval_set = [s for tpl in eval_tpls for s in by_tpl[tpl]]
    train_set = [s for tpl, items in by_tpl.items() if tpl not in eval_tpls for s in items]

    # 防泄题：两边的 template_id 必须完全不相交
    overlap = {s.template_id for s in eval_set} & {s.template_id for s in train_set}
    assert not overlap, f"评测集与训练集共享 template_id，存在泄题：{sorted(overlap)[:5]}"

    stats = {
        "eval_total": len(eval_set),
        "train_total": len(train_set),
        "eval_by_category": dict(Counter(s.category for s in eval_set)),
        "eval_by_layer": dict(Counter(layer[s.template_id] for s in eval_set)),
        "eval_by_action": dict(Counter(expected_action(s) for s in eval_set)),
    }
    return eval_set, train_set, stats


def write(eval_set, train_set, stats, by_name, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 评测题：ShareGPT 形式（供渲染 prompt）+ gold（供判分）
    with (out_dir / "eval.jsonl").open("w", encoding="utf-8") as fh:
        for s in eval_set:
            rec = to_sharegpt(s, by_name)
            rec["gold"] = gold_of(s)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 训练集的 IR，供 split.py 继续切 train/validation
    with (out_dir / "train_ir.jsonl").open("w", encoding="utf-8") as fh:
        for s in train_set:
            fh.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")

    (out_dir / "eval_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="构建评测集（按行为类别显式挑选）")
    ap.add_argument("--ir-dir", default="dataset/data/ir")
    ap.add_argument("--schemas", default="dataset/schemas/tool_schemas.json")
    ap.add_argument("--out", default="dataset/data/eval")
    ap.add_argument("--seed", type=int, default=20260810)
    args = ap.parse_args(argv)

    schemas, _ = load_schemas(args.schemas)
    by_name = schemas_by_name(schemas)
    samples = [s for f in iter_ir_files(args.ir_dir) for s in load_samples(f)]

    eval_set, train_set, stats = build(samples, seed=args.seed)
    write(eval_set, train_set, stats, by_name, Path(args.out))

    print(f"评测集 {stats['eval_total']} 条 / 训练集 {stats['train_total']} 条  → {args.out}/")
    print("\n按类别：")
    for c, n in sorted(stats["eval_by_category"].items()):
        want = QUOTA.get(c, 0)
        mark = "" if n >= want else f"  ← 不足 {want}，该行为的指标统计意义弱"
        print(f"  {c:20s} {n:4d}{mark}")
    print("\n按层：")
    for k, n in sorted(stats["eval_by_layer"].items()):
        print(f"  {k:14s} {n:4d}")
    print("\n按标准动作：")
    for k, n in sorted(stats["eval_by_action"].items()):
        print(f"  {k:22s} {n:4d}")

    missing = [c for c in QUOTA if c not in stats["eval_by_category"]]
    if missing:
        print(f"\n⚠️  这些类别在评测集里一条都没有，无法评测：{missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

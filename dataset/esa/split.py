"""按 template_id 整组切分，并渲染成 LLaMA-Factory 要的 ShareGPT jsonl。

必须整组切分：同一个模板产出的 `计算 sqrt(16)` / `sqrt(25)` / `sqrt(36)` 如果被随机
拆到 train 和 test，测试集就直接泄漏了，评测数字会好看得没有意义。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from .ir import Sample, load_samples, load_schemas, schemas_by_name
from .render import to_sharegpt


def group_split(
    samples: list[Sample],
    ratios: tuple[float, float, float] = (0.9, 0.05, 0.05),
    seed: int = 20260804,
) -> dict[str, list[Sample]]:
    """按 template_id 整组切分，并**按 category 分层**。

    只按 template 切会出问题：模板数少而每组样本多时，validation 很可能整个只落到
    一两个模板上，等于只测了一种行为。分层保证每个 category 在三个集合里都有代表，
    同时仍然不拆开任何一个 template（否则测试集直接泄漏）。
    """
    groups: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        groups[s.template_id].append(s)

    by_cat: dict[str, list[str]] = defaultdict(list)
    for tpl, items in groups.items():
        by_cat[items[0].category].append(tpl)

    rng = random.Random(seed)
    out: dict[str, list[Sample]] = {"train": [], "validation": [], "test": []}

    for cat in sorted(by_cat):
        keys = sorted(by_cat[cat])
        rng.shuffle(keys)
        cat_total = sum(len(groups[k]) for k in keys)
        want_train = cat_total * ratios[0]
        want_val = cat_total * ratios[1]

        n = 0
        for k in keys:
            if n < want_train:
                bucket = "train"
            elif n < want_train + want_val:
                bucket = "validation"
            else:
                bucket = "test"
            out[bucket].extend(groups[k])
            n += len(groups[k])

    return out


def split_warnings(split: dict[str, list[Sample]]) -> list[str]:
    """切分质量告警：某个类别在 val/test 里缺席就说明评测覆盖不全。"""
    cats = {s.category for s in split["train"]} | {s.category for s in split["validation"]} | {
        s.category for s in split["test"]
    }
    out = []
    for bucket in ("validation", "test"):
        present = {s.category for s in split[bucket]}
        for missing in sorted(cats - present):
            out.append(f"{bucket} 里没有 {missing} 类样本，该行为无法被评测到")
    return out


def assert_no_leak(split: dict[str, list[Sample]]) -> None:
    tpl = {k: {s.template_id for s in v} for k, v in split.items()}
    for a in tpl:
        for b in tpl:
            if a < b and (shared := tpl[a] & tpl[b]):
                raise AssertionError(f"{a} 与 {b} 共享 template_id，存在泄漏: {sorted(shared)[:5]}")


DATASET_INFO_KEYS = {"train": "esa_agent_train", "validation": "esa_agent_validation", "test": "esa_agent_test"}


def write_outputs(split: dict[str, list[Sample]], by_name, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {}
    for bucket, samples in split.items():
        fname = f"{DATASET_INFO_KEYS[bucket]}.jsonl"
        with (out_dir / fname).open("w", encoding="utf-8") as fh:
            for s in samples:
                fh.write(json.dumps(to_sharegpt(s, by_name), ensure_ascii=False) + "\n")
        # columns 必须显式声明 system，否则顶层 system 字段会被静默丢弃
        # （src/llamafactory/data/parser.py:81-84）
        info[DATASET_INFO_KEYS[bucket]] = {
            "file_name": fname,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system", "tools": "tools"},
        }
    (out_dir / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="切分并渲染训练数据")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--schemas", default="dataset/schemas/tool_schemas.json")
    ap.add_argument("--out", default="dataset/data/out")
    ap.add_argument("--seed", type=int, default=20260804)
    args = ap.parse_args(argv)

    schemas, _ = load_schemas(args.schemas)
    by_name = schemas_by_name(schemas)

    samples: list[Sample] = []
    for f in args.files:
        samples.extend(load_samples(f))

    split = group_split(samples, seed=args.seed)
    assert_no_leak(split)
    write_outputs(split, by_name, Path(args.out))

    from collections import Counter

    for bucket, items in split.items():
        cats = Counter(s.category for s in items)
        detail = " ".join(f"{c}={n}" for c, n in sorted(cats.items()))
        print(f"{bucket:11s} {len(items):5d} 条 / {len({s.template_id for s in items}):3d} 个模板  {detail}")

    warnings = split_warnings(split)
    if warnings:
        print()
        for w in warnings:
            print(f"⚠️  {w}")
    print(f"\n已写入 {args.out}/ （含 dataset_info.json）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

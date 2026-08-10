"""load_skill 与 RAG 的数据生成器。

load_skill —— 观测值是仓库里的**真实技能正文**（data/cache/skills_bodies.json），
  零编造。失败观测用真实的 "{name} skill not found!"。

RAG —— 只做负样本与约束样本。retrieve_knowledge 的返回依赖真实语料库，
  不知道索引了哪些文档就生成正例，等于编造可追溯来源，而「内容可追溯」
  是赛题明写的评分项。正例需要 RAG 负责人提供真实检索导出。

用法：
    python3 dataset/generators/gen_skills_rag.py
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from esa.ir import Sample, ToolCall, ToolResult, Turn, dump_samples, load_schemas  # noqa: E402
from esa.render import pick_tool_names  # noqa: E402
from esa.system_prompt import SKILL_NAMES, build_system_prompt, style_for_answer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SEEDS = ROOT / "dataset/seeds/skills_rag.yaml"
BODIES = ROOT / "dataset/data/cache/skills_bodies.json"
SCHEMAS = ROOT / "dataset/schemas/tool_schemas.json"
OUT = ROOT / "dataset/data/ir/skills_rag.jsonl"
SOURCE = "gen_skills_rag.py"


def load_bodies() -> dict[str, str]:
    """取 frontmatter 之后的正文 —— skills.load_skill 返回的是 skill.body。"""
    raw = json.loads(BODIES.read_text(encoding="utf-8"))
    out = {}
    for name, text in raw.items():
        parts = re.split(r"^---\s*$", text, maxsplit=2, flags=re.M)
        out[name] = (parts[2] if len(parts) >= 3 else text).strip()
    return out


def mk(sid, tpl, category, tools, turns, version, rng, all_names, review=False):
    return Sample(
        id=sid, template_id=tpl, category=category, schema_version=version,
        system=build_system_prompt(style=style_for_answer(turns[-1].content or "")),
        tool_names=pick_tool_names(tools, all_names, rng),
        source=SOURCE, needs_review=review, turns=turns,
    )


# 加载 Skill 之后的回应：按正文执行，而不是复述正文。
# system prompt 要求「先调用 load_skill 加载完整正文，再按正文执行」。
FOLLOW_UP = {
    "homework_review": "我按批改流程来看：先确认你的思路，再定位具体问题。你这段代码的内层循环边界 `range(n)` 会让 `a[j+1]` 越界，应该是 `range(n-1-i)`。先改这一处，再跑一遍看看？",
    "progressive_hint": "那我给你第一级提示，先不碰具体做法：想想这个问题的输入规模决定了你能用多高的复杂度——这一步能圈定算法范围。还需要下一级提示吗？",
    "error_diagnosis": "要定位根源，我需要看具体是哪一类错。你把这几次做错的题发我，或者告诉我错在哪一步——我判断一下是概念没通、步骤记错，还是前置知识的问题。",
    "teach_back": "好，你讲，我不打断。讲完我会指出哪些点说清楚了、哪些还有缺口，缺口部分我们再补。",
    "study_plan": "那我按剩余时间和你的真实掌握情况来排。先确认两件事：是哪门课，距离考试还有几周？",
    "retrieve_first": "在讲之前先做个低成本回忆——你现在对这个概念知道多少？哪怕只记得一两句也说说，我针对缺口讲会更有效率。",
}


def gen_load_skill(cfg, bodies, version, rng, all_names, out):
    for item in cfg["load_skill"]["正例"]:
        skill = item["skill"]
        body = bodies[skill]
        for j, q in enumerate(item["queries"]):
            out.append(mk(
                f"skill_{skill}_{j:02d}", f"skill__{skill}__{j:02d}", "single_tool_call",
                ["load_skill"],
                [Turn(role="user", content=q),
                 Turn(role="tool_call", calls=[ToolCall("load_skill", {"name": skill})]),
                 Turn(role="tool_result", results=[ToolResult("load_skill", body)]),
                 Turn(role="assistant", content=FOLLOW_UP[skill])],
                version, rng, all_names, review=True))

    # 不存在的 Skill —— 真实失败观测
    for j, item in enumerate(cfg["load_skill"]["不存在的skill"]):
        name = item["name"]
        err = f"{name} skill not found!"
        reply = (f"没有「{name}」这个技能。我目前可用的是："
                 f"{'、'.join(SKILL_NAMES[:5])} 等。你想做的事更接近哪一类？我不会编一个不存在的流程出来。")
        out.append(mk(
            f"skill_404_{j:02d}", f"skill__notfound__{j:02d}", "tool_error",
            ["load_skill"],
            [Turn(role="user", content=item["q"]),
             Turn(role="tool_call", calls=[ToolCall("load_skill", {"name": name})]),
             Turn(role="tool_result", results=[ToolResult("load_skill", err, is_error=True)]),
             Turn(role="assistant", content=reply)],
            version, rng, all_names))

    for group in ("混淆_不该加载", "混淆_已自动加载"):
        for j, item in enumerate(cfg["load_skill"][group]):
            out.append(mk(
                f"skill_neg_{group[-4:]}_{j:02d}", f"skill__{group}__{j:02d}", "hard_negative",
                ["load_skill", "recommend_practice", "calculator"],
                [Turn(role="user", content=item["q"]),
                 Turn(role="assistant", content=item["a"])],
                version, rng, all_names))


def gen_rag(cfg, version, rng, all_names, out):
    for j, item in enumerate(cfg["rag"]["混淆_不该检索"]):
        out.append(mk(
            f"rag_neg_{j:02d}", f"rag__不该检索__{j:02d}", "hard_negative",
            ["retrieve_knowledge", "get_knowledge_base_stats", "calculator"],
            [Turn(role="user", content=item["q"]),
             Turn(role="assistant", content=item["a"])],
            version, rng, all_names, review=True))

    for j, item in enumerate(cfg["rag"]["参数约束"]):
        out.append(mk(
            f"rag_param_{j:02d}", f"rag__参数约束__{j:02d}", "hard_negative",
            ["retrieve_knowledge", "get_knowledge_base_stats"],
            [Turn(role="user", content=item["q"]),
             Turn(role="assistant", content=item["a"])],
            version, rng, all_names))


def main() -> int:
    rng = random.Random(20260810)
    cfg = yaml.safe_load(SEEDS.read_text(encoding="utf-8"))
    bodies = load_bodies()
    schemas, version = load_schemas(SCHEMAS)
    all_names = [s["function"]["name"] for s in schemas]
    out: list[Sample] = []

    gen_load_skill(cfg, bodies, version, rng, all_names, out)
    gen_rag(cfg, version, rng, all_names, out)

    dump_samples(out, OUT)
    from collections import Counter

    print(f"生成 {len(out)} 条 → {OUT.relative_to(ROOT)}")
    for cat, n in sorted(Counter(s.category for s in out).items()):
        print(f"  {cat:20s} {n}")
    print("\n⚠️  retrieve_knowledge / get_knowledge_base_stats 只有负样本。")
    print("    正例需要 RAG 负责人提供真实语料库的检索导出 —— 不知道索引了哪些文档就造正例，")
    print("    等于编造可追溯来源，而「内容可追溯」是赛题评分项。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

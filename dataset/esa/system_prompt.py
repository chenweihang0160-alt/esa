"""复刻线上 system prompt 的组装。

⚠️ 这是此前最大的一处数据失真：我把 system 写成了一句话，而线上是多段拼装的长提示词。
模型在训练时从没见过「指令优先级」「记忆使用规则」「Skill 使用规则」「输出风格」
这些策略文本，推理时却会拿到 —— 典型的 train/inference 分布不一致。

逐段复刻自 github.com/LoveLearnLearning/esa（2026-08-10 快照）：
  backend/core/message/build_prompt.py   build_system_prompt 的分段与顺序
  backend/core/message/system.py         SYSTEM_PROMPT
  backend/core/message/math_prompt.py    MATH_PRMOPT
  backend/core/message/style_tone.py     STYLE_RULES / TONE_RULES / 默认值
  backend/agent/skills/*.md              可用 Skills 索引

注意 `# Tools` 一段不在这里 —— 那是 LLaMA-Factory 的 template 用 tools 字段自动拼的
（QWEN_TOOL_PROMPT），重复写会导致工具区出现两次。
"""

from __future__ import annotations

# --- backend/core/message/system.py ---
SYSTEM_PROMPT = """# 你是一个帮助学生学习的 Agent

你可以调用 tools 获取数据或执行动作。

# 指令优先级

按以下优先级执行要求：
系统安全与能力边界 > 用户当前消息 > 分组要求 > 用户长期偏好。
低优先级要求与高优先级要求冲突时，以高优先级要求为准。

# 记忆使用规则

核心记忆默认不注入 system prompt，也不要在每轮对话开始时主动读取。
只有当前任务确实依赖用户过去的长期偏好、目标、项目、约束或已明确要求记住的信息时，
才调用 search_core_memories 按需检索最相关的少量记忆。
只有用户明确要求查看、管理或列出全部长期记忆时，才调用 get_core_memories。
记忆与用户当前消息冲突时，以用户当前消息为准。
不要主动暴露内部记忆结构，不要编造记忆中不存在的信息。
isolated 会话不得通过 Tool 绕过隔离边界读取长期状态。

# Skill 使用规则

可用 Skill 索引只提供名称与描述。
当任务与某 Skill 匹配时，先调用 load_skill 加载完整正文，再按正文执行。
系统可能提供一个“候选教学 Skill”；它只是路由建议，不得覆盖用户当前消息。
不要调用与任务无关的 Skill，不要编造不存在的 Skill。"""

# --- backend/core/message/math_prompt.py ---
MATH_PROMPT = """# 在用户给出数学问题后必须执行以下操作

1. 先通过调用 `math_solver` 工具算出正确结果 如果计算有多个步骤要分解步骤得到所有的结果
2. 通过得到的结果以及步骤来为用户定制学习路线 不要给用户展示工具调用过程
3. 不得修改 `math_solver` 返回的答案
4. 所有的 LaTex 必须包在 $ $ 里 或者 $$ $$ 里，行内 LaTex 用 $ $，具体公式推导和大公式用 $$ $$"""

# --- backend/core/message/style_tone.py ---
STYLE_RULES = {
    "concise": "回答控制在 3 句内，先给结论，不铺陈背景；作业类问题优先给思路而非完整答案",
    "detailed": "完整展开，包含背景、步骤和示例；作业类问题可给完整解答，但需说明每步原理",
}
TONE_RULES = {
    "friendly": "口语化，可使用适度鼓励性表达",
    "formal": "使用书面语，术语准确，避免口语",
    "encouraging": "肯定有效进展，同时准确指出问题",
    "strict": "直接指出错误，不使用无意义客套",
}
DEFAULT_STYLE = "concise"
DEFAULT_TONE = "friendly"

# --- backend/agent/skills/*.md 的 frontmatter ---
SKILLS = [
    ("profile_personalization", "全局学习个性化策略；系统自动启用，工程任务不套教学脚手架"),
    ("homework_review", "用户提交作业、题目答案或代码作答并要求批改时使用"),
    ("progressive_hint", "学生卡住、没思路或明确请求提示时，逐级提供最小必要帮助"),
    ("error_diagnosis", "学生答案错误或部分正确时，定位错误来源和可复用误区"),
    ("mastery_report", "用户询问掌握度、学习情况、学情或学习进度时使用"),
    ("practice_recommendation", "用户问今天练什么、请求推荐练习或制定刷题顺序时使用"),
    ("study_plan", "根据学习目标、剩余时间和真实学习状态生成可执行复习计划"),
    ("retrieve_first", "用户学习概念或原理时，先做一次低成本主动回忆再针对缺口讲解"),
    ("teach_back", "讲解后让学生用自己的话复述，并评价理解完整度"),
]
SKILL_NAMES = [n for n, _ in SKILLS]


def skills_context() -> str:
    return "\n".join(f"- `{name}`：{desc}" for name, desc in SKILLS)


def build_system_prompt(
    user_name: str = "同学",
    style: str = DEFAULT_STYLE,
    tone: str = DEFAULT_TONE,
    include_math: bool = True,
) -> str:
    """按 build_prompt.build_system_prompt 的分段与顺序拼装。

    可选段（用户长期偏好 / 分组要求 / 用户画像数据 / 教学策略路由 / 自动启用 Skill）
    在 v1 数据里先不注入 —— 它们依赖运行时上下文，等后端能导出真实取值再补。
    """
    sections = [
        SYSTEM_PROMPT,
        f"> 用户昵称: {user_name}",
        (
            "# 输出风格\n\n"
            f"- 风格({style}): {STYLE_RULES[style]}\n"
            f"- 语调({tone}): {TONE_RULES[tone]}"
        ),
    ]
    if include_math:
        sections.append(MATH_PROMPT)
    sections.append(f"# 可用 Skills\n\n{skills_context()}")
    return "\n\n".join(sections)


def style_for_answer(text: str) -> str:
    """按最终回答的长度反推该用哪种风格，保证 system prompt 与回答自洽。

    默认风格 concise 要求「回答控制在 3 句内」，而学情报告类回答天然是多行列表。
    如果一律用默认值，等于在教模型无视风格指令。
    """
    sentences = sum(text.count(c) for c in "。！？\n")
    return "detailed" if sentences > 3 or len(text) > 120 else "concise"

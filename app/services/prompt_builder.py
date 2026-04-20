from __future__ import annotations

from dataclasses import dataclass

from app.schemas.route_and_plan import RouteAndPlanResponse


@dataclass(frozen=True)
class PromptPack:
    system_prompt: str
    user_prompt: str
    allowed_evidence_ids: list[str]


MODE_CONTRACTS = {
    "teaching": """
你现在处于 teaching 模式。
目标：把用户带到“可理解、可执行”的状态。
必须：
1. 先界定问题边界。
2. 明确分步骤推进。
3. 最后收束成可执行规则。
禁止：
- 不要只讲概念不推进
- 不要变成陪伴闲聊
- 不要变成锐评
""".strip(),
    "bridge": """
你现在处于 bridge 模式。
目标：先降低进入门槛，再把用户顺滑带进知识点。
必须：
1. 开头轻一点，不要压人。
2. 中段开始进入真正的信息。
3. 轻松表达不能变成空话。
禁止：
- 不要纯陪伴不进内容
- 不要突然变成教科书
- 不要变成锐评
""".strip(),
    "commentary": """
你现在处于 commentary 模式。
目标：直接判断，指出问题结构和本质。
必须：
1. 先给立场或判断。
2. 再拆原因、结构、后果。
3. 结尾给出明确结论。
禁止：
- 不要两边都说
- 不要温吞
- 不要只解释不判断
""".strip(),
    "persona": """
你现在处于 persona 模式。
目标：保持互动感、人设临场感和陪伴感。
必须：
1. 开头先建立互动。
2. 中间可以带内容，但不能全是硬知识。
3. 结尾给一个还能继续聊下去的钩子。
禁止：
- 不要机械陪伴
- 不要突然教科书化
- 不要变成纯评论输出
""".strip(),
}


def _clip_text(text: str, limit: int = 140) -> str:
    text = " ".join(text.strip().split())
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


class PromptBuilder:
    version = "prompt.builder.v2"

    def build(self, plan: RouteAndPlanResponse) -> PromptPack:
        mode = plan.predicted_mode.label
        shell = plan.draft_answer_shell

        skills_block = "\n".join(
            [f"- {skill.skill_id}: {skill.name}" for skill in plan.selected_skills]
        ) or "- none"

        evidence_block_lines: list[str] = []
        allowed_evidence_ids: list[str] = []
        for ev in plan.evidence_candidates[:5]:
            allowed_evidence_ids.append(ev.evidence_id)
            evidence_block_lines.append(
                f"[{ev.evidence_id}] { _clip_text(ev.text, 160) }"
            )
        evidence_block = "\n".join(evidence_block_lines) or "none"

        body_outline = "\n".join([f"- {x}" for x in shell.body_outline]) or "- none"
        do_not_do = "\n".join([f"- {x}" for x in shell.do_not_do]) or "- none"

        system_prompt = f"""
你是人物操作系统中的答案生成器。
你不能重新做路由，也不能重新选技能。
你只能基于已经给定的 mode、skills、evidence、answer shell 来生成最终答案。

{MODE_CONTRACTS.get(mode, "")}

硬约束：
1. 只能输出 JSON。
2. JSON 结构必须严格是：
{{
  "answer_text": "...",
  "used_evidence_ids": ["..."],
  "mode_check": "ok"
}}
3. used_evidence_ids 只能从给定 evidence ids 中选择。
4. 不要输出 markdown，不要输出解释，不要输出代码块。
5. 不要捏造没有提供的经历、人物背景或事实。
6. answer_text 必须是完整中文回答，而不是提纲。
7. 优先使用给定 answer shell 的 opener / body logic / closer。
""".strip()

        user_prompt = f"""
用户问题：
{plan.query}

预测模式：
{mode}

技能：
{skills_block}

答案骨架：
- opener: {shell.opener}
- body_outline:
{body_outline}
- closer: {shell.closer}
- do_not_do:
{do_not_do}

证据（只能使用这些 evidence ids）：
{evidence_block}

请严格按模式要求生成最终答案。
""".strip()

        return PromptPack(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            allowed_evidence_ids=allowed_evidence_ids,
        )
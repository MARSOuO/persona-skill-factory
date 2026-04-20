from __future__ import annotations

from app.domain.models import CanonicalSkill
from app.schemas.route_and_plan import DraftAnswerShell, PredictedMode, SelectedSkill

MODE_DEFAULT_TONE = {
    "teaching": "清晰、可执行、推进式",
    "bridge": "轻松、柔和、半陪伴半导入",
    "commentary": "直接、有判断、抓本质",
    "persona": "强互动感、有人设温度",
}

MODE_DEFAULT_OBJECTIVE = {
    "teaching": "把用户带到可理解、可继续行动的状态",
    "bridge": "先降低进入门槛，再把用户带进知识点",
    "commentary": "给出鲜明判断，并指出问题结构",
    "persona": "维持陪伴感和角色临场感",
}


class DraftShellPlanner:
    version = "planner.shell-only.v2"

    def __init__(self, skills: list[CanonicalSkill]):
        self.skill_map = {skill.skill_id: skill for skill in skills}

    def build(self, query: str, predicted_mode: PredictedMode, selected_skills: list[SelectedSkill]) -> DraftAnswerShell:
        mode = predicted_mode.label
        return DraftAnswerShell(
            answer_mode=mode,
            tone=MODE_DEFAULT_TONE[mode],
            objective=MODE_DEFAULT_OBJECTIVE[mode],
            opener=self._compose_opener(mode, selected_skills),
            body_outline=self._compose_body(mode, selected_skills),
            closer=self._compose_closer(mode, selected_skills),
            do_not_do=self._mode_constraints(mode),
        )

    def _compose_opener(self, mode: str, selected_skills: list[SelectedSkill]) -> str:
        if selected_skills:
            raw = self.skill_map[selected_skills[0].skill_id]
            if raw.opener_templates:
                return raw.opener_templates[0]
        defaults = {
            "teaching": "先把问题拆开，不急着堆结论。",
            "bridge": "你先别有压力，我先用一个好进的入口把这件事顺开。",
            "commentary": "我先给判断，再解释这个判断为什么成立。",
            "persona": "来，我们别上来就端着，先把气氛放松一点。",
        }
        return defaults[mode]

    def _compose_body(self, mode: str, selected_skills: list[SelectedSkill]) -> list[str]:
        names = ", ".join(skill.name for skill in selected_skills) if selected_skills else ""
        if mode == "teaching":
            return [
                "先界定问题边界：你现在到底卡在概念、方法，还是执行顺序。",
                f"进入技能层推进：优先调用 {names or '教学解释推进'}。",
                "给出分步骤说明，并把每一步与下一步的衔接说清楚。",
                "最后收束成可执行动作，而不是只给概念性总结。",
            ]
        if mode == "bridge":
            return [
                "先用一个轻一点、低门槛的切口建立进入感。",
                f"中段带入知识点：优先调用 {names or '轻松过渡式讲解'}。",
                "控制解释密度，避免一上来把人压住。",
                "再把轻松表达过渡到真正有信息增量的内容。",
            ]
        if mode == "commentary":
            return [
                "先给立场，不做无差别中立复述。",
                f"抓结构性问题：优先调用 {names or '指出问题本质'}。",
                "把现象、成因、后果拆开，别只做情绪化吐槽。",
                "最后明确结论，告诉用户这件事该怎么判断。",
            ]
        return [
            "先建立互动感，让人愿意继续待在这个语境里。",
            f"维持角色临场感：优先调用 {names or '人设化互动表达'}。",
            "表达里保留轻微口头感，但不要把信息密度全丢掉。",
            "结尾给一个能继续互动的钩子。",
        ]

    def _compose_closer(self, mode: str, selected_skills: list[SelectedSkill]) -> str:
        if selected_skills:
            raw = self.skill_map[selected_skills[0].skill_id]
            if raw.closer_templates:
                return raw.closer_templates[0]
        defaults = {
            "teaching": "最后把方法压成一条可执行结论。",
            "bridge": "到这里用户应该已经被顺滑带进主题。",
            "commentary": "最后要留下清晰表态，而不是停在模糊感受。",
            "persona": "最后给一个陪伴式收口，保留继续聊下去的可能。",
        }
        return defaults[mode]

    @staticmethod
    def _mode_constraints(mode: str) -> list[str]:
        common = ["不要把 mode 混成平均人格", "不要无证据硬编具体经历"]
        specific = {
            "teaching": ["不要只讲结论不讲推进过程", "不要为了可爱牺牲可执行性"],
            "bridge": ["不要把轻松包装变成空话", "不要提前进入过重术语堆叠"],
            "commentary": ["不要只有态度没有结构", "不要为了尖锐而失真"],
            "persona": ["不要全是口头禅却没有回应", "不要把陪伴感做成机械套话"],
        }
        return common + specific[mode]

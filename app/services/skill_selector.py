from __future__ import annotations

from app.domain.models import CanonicalSkill
from app.schemas.route_and_plan import SelectedSkill
from app.services.router import RouterService

GENERIC_SKILL_IDS = {
    "teaching.explain_progression",
}

BROAD_TEACHING_SIGNALS = {
    "讲明白", "讲清楚", "慢慢讲", "带着我", "一步一步", "解释清楚",
}

SPECIFIC_TEACHING_SIGNALS = {
    "分步骤", "总结", "归纳", "举例", "例子", "类比", "先判断", "再选方法", "选方法",
}

STANCE_STRONG_SIGNALS = {
    "表态", "明确表态", "站队", "给立场", "说你的立场",
    "旗帜鲜明", "别模棱两可", "不要中立", "直接站哪边",
    "明确支持", "明确反对", "站哪边", "亮明立场",
}

STANCE_WEAK_SIGNALS = {
    "你的判断", "你怎么看", "直接说判断", "直接说你的判断", "说结论", "别只复述",
}


class SkillSelectorService:
    version = "selector.primary-support.v4"

    def __init__(self, skills: list[CanonicalSkill]):
        self.skills = skills

    def select(self, query: str, mode: str, top_k: int) -> list[SelectedSkill]:
        q = query.strip().lower()
        tokens = set(RouterService.tokenize(q))
        candidates: list[SelectedSkill] = []

        broad_teaching = any(x in q for x in BROAD_TEACHING_SIGNALS)
        specific_teaching = any(x in q for x in SPECIFIC_TEACHING_SIGNALS)

        strong_stance = any(x in q for x in STANCE_STRONG_SIGNALS)
        weak_stance = any(x in q for x in STANCE_WEAK_SIGNALS)

        mode_skills = [s for s in self.skills if s.mode == mode]

        for skill in mode_skills:
            score = 0.0
            reasons: list[str] = []

            trigger_hits = [t for t in skill.triggers if t.lower() in q or t.lower() in tokens]
            anti_hits = [t for t in skill.anti_triggers if t.lower() in q or t.lower() in tokens]
            summary_hits = [t for t in RouterService.tokenize(skill.summary) if t in tokens]

            if trigger_hits:
                score += 2.0 * len(trigger_hits)
                reasons.append(f"触发词匹配: {', '.join(trigger_hits[:4])}")

            if summary_hits:
                score += 0.3 * len(summary_hits)
                reasons.append("query 与 skill 摘要存在语义重合")

            if anti_hits:
                score -= 1.25 * len(anti_hits)
                reasons.append(f"命中反触发词: {', '.join(anti_hits[:3])}")

            if skill.skill_id not in GENERIC_SKILL_IDS and trigger_hits:
                score += 0.4
                reasons.append("具体 skill 命中，给予具体性加分")

            # teaching: generic skill 去冗余
            if skill.skill_id == "teaching.explain_progression":
                if specific_teaching:
                    score -= 1.2
                    reasons.append("去冗余: query 已指向更具体 teaching 技能，通用推进下调")
                elif broad_teaching:
                    score += 0.8
                    reasons.append("query 明确要求整体讲明白，通用推进上调")

            # commentary: judgment != explicit stance
            if skill.skill_id == "commentary.explicit_stance":
                if strong_stance:
                    score += 1.2
                    reasons.append("明确立场信号强，表态技能上调")
                elif weak_stance:
                    score -= 1.0
                    reasons.append("仅要求判断而非表态，明确表态技能下调")
                else:
                    score -= 0.6
                    reasons.append("未出现强立场信号，明确表态技能轻度下调")

            if skill.skill_id == "commentary.judgment_output":
                if weak_stance or strong_stance:
                    score += 0.8
                    reasons.append("query 明确要求给判断，判断输出上调")

            if not trigger_hits and not summary_hits:
                score += 0.05
                reasons.append("同 mode 低权重兜底")

            candidates.append(
                SelectedSkill(
                    skill_id=skill.skill_id,
                    mode=skill.mode,
                    name=skill.name,
                    score=round(score, 4),
                    reasons=reasons,
                )
            )

        candidates.sort(key=lambda item: item.score, reverse=True)
        if not candidates:
            return []

        primary = candidates[0]
        result = [primary]

        if top_k <= 1 or len(candidates) == 1:
            return result

        second = candidates[1]

        score_gap = primary.score - second.score
        ratio = second.score / max(primary.score, 1e-6)

        keep_second = False

        if ratio >= 0.72 and score_gap <= 1.4:
            keep_second = True

        # teaching: generic second 去冗余
        if (
            second.skill_id in GENERIC_SKILL_IDS
            and primary.skill_id not in GENERIC_SKILL_IDS
            and not broad_teaching
        ):
            keep_second = False

        # teaching: primary/secondary swap
        if (
            primary.skill_id in GENERIC_SKILL_IDS
            and second.skill_id not in GENERIC_SKILL_IDS
            and ratio >= 0.85
        ):
            primary, second = second, primary
            result = [primary]
            keep_second = False if second.skill_id in GENERIC_SKILL_IDS and specific_teaching else keep_second

        # commentary: explicit_stance 只有强立场信号下才作为 second 保留
        if (
            mode == "commentary"
            and second.skill_id == "commentary.explicit_stance"
            and primary.skill_id == "commentary.judgment_output"
            and not strong_stance
        ):
            keep_second = False

        # commentary: 如果 primary 是 explicit_stance，但只是弱判断语气，则尝试让 judgment_output 前置
        if (
            mode == "commentary"
            and primary.skill_id == "commentary.explicit_stance"
            and second.skill_id == "commentary.judgment_output"
            and (weak_stance and not strong_stance)
        ):
            primary, second = second, primary
            result = [primary]
            keep_second = False

        if keep_second:
            result.append(second)

        return result[:top_k]

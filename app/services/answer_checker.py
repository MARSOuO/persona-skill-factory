from __future__ import annotations

from dataclasses import dataclass, field

from app.schemas.route_and_plan import RouteAndPlanResponse
from app.services.answer_generator import GeneratedAnswer


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    has_mode_drift: bool = False
    has_empty_grounding: bool = False


class AnswerChecker:
    version = "answer.check.v2"

    def check(
        self,
        plan: RouteAndPlanResponse,
        generated: GeneratedAnswer,
        allowed_evidence_ids: list[str],
    ) -> CheckResult:
        reasons: list[str] = []
        text = generated.answer_text.strip()

        if not text:
            reasons.append("empty_answer")

        valid_used_ids = [x for x in generated.used_evidence_ids if x in allowed_evidence_ids]
        has_empty_grounding = len(valid_used_ids) == 0
        if has_empty_grounding:
            reasons.append("empty_grounding")

        has_mode_drift = self._has_mode_drift(plan.predicted_mode.label, text)
        if has_mode_drift:
            reasons.append("mode_drift")

        ok = bool(text) and (not has_empty_grounding) and (not has_mode_drift)

        return CheckResult(
            ok=ok,
            reasons=reasons,
            has_mode_drift=has_mode_drift,
            has_empty_grounding=has_empty_grounding,
        )

    def _has_mode_drift(self, mode: str, text: str) -> bool:
        if mode == "teaching":
            return not any(k in text for k in ["先", "再", "最后", "步骤", "第一", "第二"])

        if mode == "bridge":
            return not any(k in text for k in ["先别急", "先不用", "顺着", "慢慢", "带你", "先把它"])

        if mode == "commentary":
            has_judgment = any(k in text for k in ["我先直接", "判断", "本质", "问题在于", "所以"])
            has_both_sides = any(k in text for k in ["一方面", "另一方面", "都对", "都可以"])
            return (not has_judgment) or has_both_sides

        if mode == "persona":
            return not any(k in text for k in ["你", "我", "咱", "跟我", "我们"])

        return False
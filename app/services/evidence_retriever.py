from __future__ import annotations

from app.domain.models import CanonicalSkill
from app.schemas.route_and_plan import EvidenceCandidate, SelectedSkill
from app.services.router import RouterService


class EvidenceRetrieverService:
    version = "evidence.canonical-bound.v2"

    def __init__(self, skills: list[CanonicalSkill]):
        self.skill_map = {skill.skill_id: skill for skill in skills}

    def retrieve(self, query: str, selected_skills: list[SelectedSkill], top_k: int) -> list[EvidenceCandidate]:
        tokens = RouterService.tokenize(query)
        results: list[EvidenceCandidate] = []

        for skill in selected_skills:
            raw_skill = self.skill_map[skill.skill_id]
            for index, text in enumerate(raw_skill.evidence_examples, start=1):
                hit_bonus = 0.0
                lower_text = text.lower()
                for token in tokens:
                    if token and token in lower_text:
                        hit_bonus += 0.15
                results.append(
                    EvidenceCandidate(
                        evidence_id=f"{skill.skill_id}::e{index}",
                        skill_id=skill.skill_id,
                        mode=skill.mode,
                        text=text,
                        score=round(skill.score + hit_bonus, 4),
                    )
                )

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

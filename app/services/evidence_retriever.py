from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from app.adapters.evidence_assets import load_evidence_index, load_skill_unit_links
from app.domain.models import CanonicalSkill, EvidenceUnit
from app.schemas.route_and_plan import EvidenceCandidate, SelectedSkill
from app.services.router import RouterService


class EvidenceRetrieverService:
    version = "evidence.skill-aware-retrieval.v1"

    def __init__(
        self,
        skills: list[CanonicalSkill],
        evidence_index_path: str,
        skill_unit_links_path: str,
    ):
        self.skill_map = {skill.skill_id: skill for skill in skills}
        self.units = load_evidence_index(evidence_index_path)
        self.skill_unit_links = load_skill_unit_links(skill_unit_links_path)
        self.units_by_mode: dict[str, list[EvidenceUnit]] = defaultdict(list)
        self.unit_map: dict[str, EvidenceUnit] = {}
        for unit in self.units:
            self.units_by_mode[unit.mode].append(unit)
            self.unit_map[unit.unit_id] = unit

    def _candidate_pool(self, mode: str, selected_skills: list[SelectedSkill]) -> list[EvidenceUnit]:
        mode_units = list(self.units_by_mode.get(mode, []))
        linked_unit_ids: set[str] = set()
        for skill in selected_skills:
            linked_unit_ids.update(self.skill_unit_links.get(skill.skill_id, []))

        linked_units = [self.unit_map[u] for u in linked_unit_ids if u in self.unit_map]

        # 去重，且优先把 linked units 放在前面
        seen: set[str] = set()
        ordered: list[EvidenceUnit] = []
        for unit in linked_units + mode_units:
            if unit.unit_id not in seen:
                ordered.append(unit)
                seen.add(unit.unit_id)
        return ordered

    def retrieve(
        self,
        query: str,
        mode: str,
        selected_skills: list[SelectedSkill],
        top_k: int,
    ) -> list[EvidenceCandidate]:
        query_tokens = [t for t in RouterService.tokenize(query) if t]
        selected_skill_ids = [skill.skill_id for skill in selected_skills]
        pool = self._candidate_pool(mode, selected_skills)
        results: list[EvidenceCandidate] = []

        for unit in pool:
            lower_text = unit.text.lower()
            matched_terms = [token for token in query_tokens if token in lower_text]
            if not matched_terms and unit.mode != mode:
                continue

            score = 0.0
            if unit.mode == mode:
                score += 1.2

            score += min(1.8, 0.22 * len(set(matched_terms)))

            matched_skill_ids: list[str] = []
            best_skill_id = selected_skill_ids[0] if selected_skill_ids else ""
            best_skill_score = -1.0

            for rank, skill in enumerate(selected_skills, start=1):
                skill_bonus = 0.0

                if unit.unit_id in self.skill_unit_links.get(skill.skill_id, []):
                    skill_bonus += max(2.4 - 0.18 * rank, 1.2)

                raw_skill = self.skill_map.get(skill.skill_id)
                if raw_skill:
                    trigger_hits = [t for t in raw_skill.triggers if t and t.lower() in lower_text]
                    anti_hits = [t for t in raw_skill.anti_triggers if t and t.lower() in lower_text]
                    skill_bonus += min(1.2, 0.35 * len(trigger_hits))
                    skill_bonus -= min(0.8, 0.25 * len(anti_hits))
                    if trigger_hits or unit.unit_id in self.skill_unit_links.get(skill.skill_id, []):
                        matched_skill_ids.append(skill.skill_id)

                if skill_bonus > best_skill_score:
                    best_skill_score = skill_bonus
                    best_skill_id = skill.skill_id

                score += skill_bonus

            if not matched_skill_ids and selected_skill_ids:
                matched_skill_ids = [best_skill_id]

            results.append(
                EvidenceCandidate(
                    evidence_id=unit.evidence_id,
                    unit_id=unit.unit_id,
                    skill_id=best_skill_id,
                    matched_skill_ids=matched_skill_ids,
                    mode=unit.mode,
                    text=unit.text,
                    score=round(score, 4),
                    source="retrieval.skill-aware.v1",
                    source_file=unit.source_file,
                    paragraph_id=unit.paragraph_id,
                    matched_terms=sorted(set(matched_terms))[:8],
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

from __future__ import annotations

from collections import defaultdict

from app.adapters.evidence_assets import load_evidence_index, load_skill_unit_links
from app.domain.models import CanonicalSkill, EvidenceUnit
from app.schemas.route_and_plan import EvidenceCandidate, SelectedSkill
from app.utils.tokenize import tokenize_zh_en


class EvidenceRetrieverService:
    version = "evidence.skill-aware-rerank.v1_1b"

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

    def _build_query_terms(self, query: str) -> list[str]:
        return tokenize_zh_en(query)

    def _unit_token_set(self, unit: EvidenceUnit) -> set[str]:
        meta = unit.metadata or {}
        token_candidates = meta.get("token_set") or meta.get("unit_tokens") or []

        if isinstance(token_candidates, list):
            tokens = [str(x).lower() for x in token_candidates if str(x).strip()]
            if tokens:
                return set(tokens)

        return set(tokenize_zh_en(unit.text))

    def _unit_skill_hints(self, unit: EvidenceUnit) -> list[str]:
        meta = unit.metadata or {}
        hints = meta.get("skill_hints") or []
        if isinstance(hints, list):
            return [str(x).strip() for x in hints if str(x).strip()]
        return []

    def _candidate_pool(
        self,
        query: str,
        mode: str,
        selected_skills: list[SelectedSkill],
        top_n_per_skill: int = 12,
        lexical_top_n: int = 40,
        fallback_top_n: int = 5,
    ) -> list[EvidenceUnit]:
        mode_units = list(self.units_by_mode.get(mode, []))
        query_terms = self._build_query_terms(query)
        query_term_set = set(query_terms)

        linked_units: list[EvidenceUnit] = []
        for skill in selected_skills:
            for unit_id in self.skill_unit_links.get(skill.skill_id, [])[:top_n_per_skill]:
                unit = self.unit_map.get(unit_id)
                if unit and unit.mode == mode:
                    linked_units.append(unit)

        lexical_scored: list[tuple[int, EvidenceUnit]] = []
        for unit in mode_units:
            overlap = len(query_term_set & self._unit_token_set(unit))
            if overlap > 0:
                lexical_scored.append((overlap, unit))

        lexical_scored.sort(key=lambda x: x[0], reverse=True)
        lexical_units = [unit for _, unit in lexical_scored[:lexical_top_n]]

        fallback_units = mode_units[:fallback_top_n]

        seen: set[str] = set()
        ordered: list[EvidenceUnit] = []
        for unit in linked_units + lexical_units + fallback_units:
            if unit.unit_id not in seen:
                ordered.append(unit)
                seen.add(unit.unit_id)

        return ordered

    def _apply_query_relevance_gate(
        self,
        unit: EvidenceUnit,
        query_terms: list[str],
        mode: str,
        selected_skill_ids: list[str],
    ) -> tuple[list[str], float, bool]:
        unit_tokens = self._unit_token_set(unit)
        matched_terms = sorted(set(query_terms) & unit_tokens)

        skill_hints = self._unit_skill_hints(unit)
        skill_overlap = [s for s in skill_hints if s in selected_skill_ids]

        penalty = 0.0
        topic_drift_flag = False

        if mode in {"teaching", "commentary"}:
            if not matched_terms and not skill_overlap:
                penalty -= 1.2
                topic_drift_flag = True
            elif not matched_terms and skill_overlap:
                penalty -= 0.6
                topic_drift_flag = True
        elif mode in {"bridge", "persona"}:
            if not matched_terms and not skill_overlap:
                penalty -= 1.2
                topic_drift_flag = True
            elif not matched_terms and skill_overlap:
                penalty -= 0.6
                topic_drift_flag = True

        return matched_terms, penalty, topic_drift_flag

    def _rerank_candidates(
        self,
        query: str,
        mode: str,
        selected_skills: list[SelectedSkill],
        pool: list[EvidenceUnit],
        top_k: int,
    ) -> list[EvidenceCandidate]:
        query_terms = self._build_query_terms(query)
        selected_skill_ids = [skill.skill_id for skill in selected_skills]

        results: list[EvidenceCandidate] = []

        for unit in pool:
            lower_text = unit.text.lower()
            matched_terms, penalty, topic_drift_flag = self._apply_query_relevance_gate(
                unit=unit,
                query_terms=query_terms,
                mode=mode,
                selected_skill_ids=selected_skill_ids,
            )

            score_skill = 0.0
            score_query_overlap = 0.6 * len(matched_terms)
            score_mode_bonus = 1.0 if unit.mode == mode else 0.0

            matched_skill_ids: list[str] = []
            best_skill_id = selected_skill_ids[0] if selected_skill_ids else ""
            best_skill_score = -999.0

            unit_skill_hints = self._unit_skill_hints(unit)

            for rank, skill in enumerate(selected_skills, start=1):
                skill_bonus = 0.0

                if unit.unit_id in self.skill_unit_links.get(skill.skill_id, []):
                    skill_bonus += max(1.8 - 0.15 * rank, 1.0)

                if skill.skill_id in unit_skill_hints:
                    skill_bonus += 0.8

                raw_skill = self.skill_map.get(skill.skill_id)
                if raw_skill:
                    trigger_hits = [t for t in raw_skill.triggers if t and t.lower() in lower_text]
                    anti_hits = [t for t in raw_skill.anti_triggers if t and t.lower() in lower_text]

                    skill_bonus += min(1.2, 0.35 * len(trigger_hits))
                    skill_bonus -= min(0.8, 0.25 * len(anti_hits))

                    if (
                        trigger_hits
                        or unit.unit_id in self.skill_unit_links.get(skill.skill_id, [])
                        or skill.skill_id in unit_skill_hints
                    ):
                        matched_skill_ids.append(skill.skill_id)

                if skill_bonus > best_skill_score:
                    best_skill_score = skill_bonus
                    best_skill_id = skill.skill_id

                score_skill += max(skill_bonus, 0.0)

            if not matched_skill_ids and selected_skill_ids:
                matched_skill_ids = [best_skill_id]

            if mode in {"bridge", "persona"}:
                score_total = (
                    1.6 * score_skill
                    + 3.0 * score_query_overlap
                    + 0.8 * score_mode_bonus
                    + penalty
                )
            else:
                score_total = (
                    2.5 * score_skill
                    + 1.5 * score_query_overlap
                    + 0.8 * score_mode_bonus
                    + penalty
                )

            results.append(
                EvidenceCandidate(
                    evidence_id=unit.evidence_id,
                    unit_id=unit.unit_id,
                    skill_id=best_skill_id,
                    matched_skill_ids=matched_skill_ids,
                    mode=unit.mode,
                    text=unit.text,
                    source="retrieval.skill-aware-rerank.v1_1b",
                    source_file=unit.source_file,
                    paragraph_id=unit.paragraph_id,
                    matched_terms=matched_terms[:8],
                    score=round(score_total, 4),
                    score_total=round(score_total, 4),
                    score_skill=round(score_skill, 4),
                    score_query_overlap=round(score_query_overlap, 4),
                    score_mode_bonus=round(score_mode_bonus, 4),
                    score_penalty=round(penalty, 4),
                    retrieval_stage="reranked",
                    topic_drift_flag=topic_drift_flag,
                )
            )

        if mode in {"bridge", "persona"}:
            preferred: list[EvidenceCandidate] = []
            fallback: list[EvidenceCandidate] = []

            for item in results:
                if item.matched_terms and not item.topic_drift_flag:
                    preferred.append(item)
                else:
                    fallback.append(item)

            preferred.sort(key=lambda x: x.score_total, reverse=True)
            fallback.sort(key=lambda x: x.score_total, reverse=True)
            results = preferred + fallback
        else:
            results.sort(key=lambda x: x.score_total, reverse=True)

        return results[:top_k]

    def retrieve(
        self,
        query: str,
        mode: str,
        selected_skills: list[SelectedSkill],
        top_k: int,
    ) -> list[EvidenceCandidate]:
        pool = self._candidate_pool(
            query=query,
            mode=mode,
            selected_skills=selected_skills,
        )
        return self._rerank_candidates(
            query=query,
            mode=mode,
            selected_skills=selected_skills,
            pool=pool,
            top_k=top_k,
        )

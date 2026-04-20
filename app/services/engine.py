from __future__ import annotations

from app.adapters.assets import load_skills
from app.domain.models import AppConfig
from app.schemas.route_and_plan import RouteAndPlanRequest, RouteAndPlanResponse, TraceInfo
from app.services.evidence_retriever import EvidenceRetrieverService
from app.services.planner import DraftShellPlanner
from app.services.router import RouterService
from app.services.skill_selector import SkillSelectorService


class PersonaSkillEngine:
    def __init__(self, config: AppConfig):
        self.config = config
        self.skills = load_skills(str(config.canonical_skills_path))
        self.router = RouterService(self.skills)
        self.selector = SkillSelectorService(self.skills)
        self.retriever = EvidenceRetrieverService(self.skills)
        self.planner = DraftShellPlanner(self.skills)

    def route_and_plan(self, req: RouteAndPlanRequest) -> RouteAndPlanResponse:
        normalized_query = " ".join(req.query.strip().split())
        predicted_mode = self.router.predict(normalized_query)
        selected_skills = self.selector.select(normalized_query, predicted_mode.label, req.top_k_skills)
        evidence_candidates = self.retriever.retrieve(normalized_query, selected_skills, req.top_k_evidence)
        draft_answer_shell = self.planner.build(normalized_query, predicted_mode, selected_skills)

        return RouteAndPlanResponse(
            query=req.query,
            normalized_query=normalized_query,
            predicted_mode=predicted_mode,
            selected_skills=selected_skills,
            evidence_candidates=evidence_candidates,
            draft_answer_shell=draft_answer_shell,
            trace=TraceInfo(
                asset_version=self.config.asset_version,
                selector_version=self.selector.version,
                evidence_version=self.retriever.version,
                planner_version=self.planner.version,
            ),
        )

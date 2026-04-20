from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class RouteAndPlanRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k_skills: int = Field(default=2, ge=1, le=5)
    top_k_evidence: int = Field(default=4, ge=1, le=10)
    allow_llm_router: bool = Field(default=False)
    allow_llm_draft: bool = Field(default=False)
    user_context: Optional[dict[str, Any]] = None


class ModeScore(BaseModel):
    mode: str
    score: float
    reasons: list[str] = Field(default_factory=list)


class PredictedMode(BaseModel):
    label: str
    confidence: float
    candidates: list[ModeScore]
    router_version: str


class SelectedSkill(BaseModel):
    skill_id: str
    mode: str
    name: str
    score: float
    reasons: list[str] = Field(default_factory=list)


class EvidenceCandidate(BaseModel):
    evidence_id: str
    skill_id: str
    mode: str
    text: str
    score: float
    source: str = "canonical_skill_bound_evidence"


class DraftAnswerShell(BaseModel):
    answer_mode: str
    tone: str
    objective: str
    opener: str
    body_outline: list[str]
    closer: str
    do_not_do: list[str]


class TraceInfo(BaseModel):
    asset_version: str
    selector_version: str
    evidence_version: str
    planner_version: str


class RouteAndPlanResponse(BaseModel):
    query: str
    normalized_query: str
    predicted_mode: PredictedMode
    selected_skills: list[SelectedSkill]
    evidence_candidates: list[EvidenceCandidate]
    draft_answer_shell: DraftAnswerShell
    trace: TraceInfo

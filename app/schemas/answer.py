from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.route_and_plan import RouteAndPlanResponse


class AnswerRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k_skills: int = Field(default=2, ge=1, le=5)
    top_k_evidence: int = Field(default=5, ge=1, le=10)

    allow_llm_router: bool = False
    allow_llm_draft: bool = False
    allow_llm_answer: bool = False

    return_plan: bool = True
    max_new_tokens: int = Field(default=384, ge=64, le=2048)
    temperature: float = Field(default=0.4, ge=0.0, le=1.5)


class GroundingSummary(BaseModel):
    evidence_used_count: int = 0
    has_mode_drift: bool = False
    has_empty_grounding: bool = False
    check_reasons: list[str] = Field(default_factory=list)


class AnswerTrace(BaseModel):
    asset_version: str
    selector_version: str
    evidence_version: str
    planner_version: str
    answer_version: str


class AnswerResponse(BaseModel):
    query: str
    answer_mode: str
    answer_text: str
    used_evidence_ids: list[str] = Field(default_factory=list)
    generation_mode: str
    grounding_summary: GroundingSummary
    plan: RouteAndPlanResponse | None = None
    trace: AnswerTrace

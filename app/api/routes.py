from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from app.core.dependencies import get_engine
from app.schemas.route_and_plan import RouteAndPlanRequest, RouteAndPlanResponse
from app.schemas.answer import AnswerRequest, AnswerResponse, GroundingSummary, AnswerTrace
from app.schemas.route_and_plan import RouteAndPlanRequest
from app.services.answer_checker import AnswerChecker
from app.services.answer_generator import AnswerGenerator
from app.services.prompt_builder import PromptBuilder


router = APIRouter()


@router.get("/healthz")
def healthz() -> dict[str, Any]:
    engine = get_engine()
    return {
        "ok": True,
        "app": engine.config.app_name,
        "version": engine.config.app_version,
        "skills_loaded": len(engine.skills),
        "asset_version": engine.config.asset_version,
        "canonical_skills_path": str(engine.config.canonical_skills_path),
    }


@router.get("/debug/skills")
def debug_skills() -> dict[str, Any]:
    engine = get_engine()
    return {
        "count": len(engine.skills),
        "skills": [
            {
                "skill_id": skill.skill_id,
                "mode": skill.mode,
                "name": skill.name,
                "summary": skill.summary,
            }
            for skill in engine.skills
        ],
    }


@router.post("/route_and_plan", response_model=RouteAndPlanResponse)
def route_and_plan(req: RouteAndPlanRequest) -> RouteAndPlanResponse:
    engine = get_engine()
    return engine.route_and_plan(req)

@router.post("/answer", response_model=AnswerResponse)
def answer(payload: AnswerRequest, engine=Depends(get_engine)) -> AnswerResponse:
    plan_request = RouteAndPlanRequest(
        query=payload.query,
        top_k_skills=payload.top_k_skills,
        top_k_evidence=payload.top_k_evidence,
        allow_llm_router=payload.allow_llm_router,
        allow_llm_draft=payload.allow_llm_draft,
        user_context=None,
    )

    plan = engine.route_and_plan(plan_request)

    builder = PromptBuilder()
    generator = AnswerGenerator()
    checker = AnswerChecker()

    prompt_pack = builder.build(plan)

    generated = generator.generate(
        plan=plan,
        prompt_pack=prompt_pack,
        allow_llm_answer=payload.allow_llm_answer,
        max_new_tokens=payload.max_new_tokens,
        temperature=payload.temperature,
    )

    allowed_evidence_ids = [ev.evidence_id for ev in plan.evidence_candidates]
    check = checker.check(
        plan=plan,
        generated=generated,
        allowed_evidence_ids=allowed_evidence_ids,
    )

    if not check.ok:
        generated = generator.fallback(plan)
        check = checker.check(
            plan=plan,
            generated=generated,
            allowed_evidence_ids=allowed_evidence_ids,
        )

    return AnswerResponse(
        query=payload.query,
        answer_mode=plan.predicted_mode.label,
        answer_text=generated.answer_text,
        used_evidence_ids=generated.used_evidence_ids,
        generation_mode=generated.generation_mode,
        grounding_summary=GroundingSummary(
            evidence_used_count=len(generated.used_evidence_ids),
            has_mode_drift=check.has_mode_drift,
            has_empty_grounding=check.has_empty_grounding,
            check_reasons=check.reasons,
        ),
        plan=plan if payload.return_plan else None,
        trace=AnswerTrace(
            asset_version=plan.trace.asset_version,
            selector_version=plan.trace.selector_version,
            evidence_version=plan.trace.evidence_version,
            planner_version=plan.trace.planner_version,
            answer_version=generator.version,
        ),
    )
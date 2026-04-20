from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.core.dependencies import get_engine
from app.schemas.route_and_plan import RouteAndPlanRequest, RouteAndPlanResponse

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

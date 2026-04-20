from __future__ import annotations

import os
from pathlib import Path

from app.domain.models import AppConfig

APP_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SKILLS_PATH = APP_ROOT / "assets" / "canonical_skills.v1.yaml"


def get_config() -> AppConfig:
    raw_path = os.getenv("CANONICAL_SKILLS_PATH", str(DEFAULT_SKILLS_PATH))
    return AppConfig(
        canonical_skills_path=Path(raw_path),
        asset_version=os.getenv("ASSET_VERSION", "v1-online-loop"),
        app_name=os.getenv("APP_NAME", "persona-skill-factory"),
        app_version=os.getenv("APP_VERSION", "0.2.0"),
        default_top_k_skills=int(os.getenv("DEFAULT_TOP_K_SKILLS", "2")),
        default_top_k_evidence=int(os.getenv("DEFAULT_TOP_K_EVIDENCE", "4")),
        enable_vllm_router=os.getenv("ENABLE_VLLM_ROUTER", "0") == "1",
        enable_vllm_draft=os.getenv("ENABLE_VLLM_DRAFT", "0") == "1",
        vllm_base_url=os.getenv("VLLM_BASE_URL"),
        vllm_model=os.getenv("VLLM_MODEL"),
    )

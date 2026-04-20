from __future__ import annotations

import os
from pathlib import Path

from app.domain.models import AppConfig


APP_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SKILLS_PATH = APP_ROOT / "assets" / "canonical_skills.v1.yaml"
DEFAULT_EVIDENCE_INDEX_PATH = APP_ROOT / "artifacts" / "retrieval" / "evidence_index.v1_1.jsonl"
DEFAULT_SKILL_UNIT_LINKS_PATH = APP_ROOT / "artifacts" / "retrieval" / "skill_to_unit_links.v1_1.json"


def get_config() -> AppConfig:
    raw_path = os.getenv("CANONICAL_SKILLS_PATH", str(DEFAULT_SKILLS_PATH))
    raw_evidence_index = os.getenv("EVIDENCE_INDEX_PATH", str(DEFAULT_EVIDENCE_INDEX_PATH))
    raw_skill_links = os.getenv("SKILL_UNIT_LINKS_PATH", str(DEFAULT_SKILL_UNIT_LINKS_PATH))

    return AppConfig(
        canonical_skills_path=Path(raw_path),
        evidence_index_path=Path(raw_evidence_index),
        skill_unit_links_path=Path(raw_skill_links),
        asset_version=os.getenv("ASSET_VERSION", "v1-online-loop+retrieval-v1_1"),
        app_name=os.getenv("APP_NAME", "persona-skill-factory"),
        app_version=os.getenv("APP_VERSION", "0.3.0"),
        default_top_k_skills=int(os.getenv("DEFAULT_TOP_K_SKILLS", "2")),
        default_top_k_evidence=int(os.getenv("DEFAULT_TOP_K_EVIDENCE", "4")),
        enable_vllm_router=os.getenv("ENABLE_VLLM_ROUTER", "0") == "1",
        enable_vllm_draft=os.getenv("ENABLE_VLLM_DRAFT", "0") == "1",
        vllm_base_url=os.getenv("VLLM_BASE_URL"),
        vllm_model=os.getenv("VLLM_MODEL"),
    )
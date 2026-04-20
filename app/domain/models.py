from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class CanonicalSkill:
    skill_id: str
    mode: str
    name: str
    summary: str
    triggers: list[str]
    anti_triggers: list[str]
    opener_templates: list[str]
    body_templates: list[str]
    closer_templates: list[str]
    evidence_examples: list[str]


@dataclass(frozen=True)
class AppConfig:
    canonical_skills_path: Path
    asset_version: str
    app_name: str
    app_version: str
    default_top_k_skills: int
    default_top_k_evidence: int
    enable_vllm_router: bool = False
    enable_vllm_draft: bool = False
    vllm_base_url: Optional[str] = None
    vllm_model: Optional[str] = None

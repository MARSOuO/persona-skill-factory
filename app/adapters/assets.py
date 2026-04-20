from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from app.domain.models import CanonicalSkill


@lru_cache(maxsize=8)
def load_skills(path_str: str) -> list[CanonicalSkill]:
    path = Path(path_str)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    skills: list[CanonicalSkill] = []
    for item in data.get("skills", []):
        skills.append(
            CanonicalSkill(
                skill_id=item["skill_id"],
                mode=item["mode"],
                name=item["name"],
                summary=item["summary"],
                triggers=item.get("triggers", []),
                anti_triggers=item.get("anti_triggers", []),
                opener_templates=item.get("opener_templates", []),
                body_templates=item.get("body_templates", []),
                closer_templates=item.get("closer_templates", []),
                evidence_examples=item.get("evidence_examples", []),
            )
        )
    return skills

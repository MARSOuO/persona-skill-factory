from __future__ import annotations

from functools import lru_cache

from app.core.config import get_config
from app.services.engine import PersonaSkillEngine


@lru_cache(maxsize=1)
def get_engine() -> PersonaSkillEngine:
    return PersonaSkillEngine(get_config())

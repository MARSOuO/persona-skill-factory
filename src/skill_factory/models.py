from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ModeLabel(str, Enum):
    persona = "persona"
    commentary = "commentary"
    teaching = "teaching"
    bridge = "bridge"
    unknown = "unknown"


class CleanSegment(BaseModel):
    source_file: str
    corpus_mode: ModeLabel
    segment_id: str
    text: str
    speaker: str | None = None
    start_sec: float | None = None
    end_sec: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceSpan(BaseModel):
    source_file: str
    segment_id: str
    start_char: int | None = None
    end_char: int | None = None
    start_sec: float | None = None
    end_sec: float | None = None


class UnitRecord(BaseModel):
    unit_id: str
    corpus_mode: ModeLabel
    router_mode: ModeLabel
    source: SourceSpan
    text: str
    normalized_text: str
    style_signals: list[str] = Field(default_factory=list)
    teaching_signals: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    extractable: bool = True
    confidence: float = 0.8
    metadata: dict[str, Any] = Field(default_factory=dict)


class CandidateSkillRecord(BaseModel):
    skill_id: str
    name: str
    mode: ModeLabel
    description: str
    trigger: list[str] = Field(default_factory=list)
    diagnosis: list[str] = Field(default_factory=list)
    response_plan: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    negative_cases: list[str] = Field(default_factory=list)
    evidence_units: list[str] = Field(default_factory=list)
    evidence_quotes: list[str] = Field(default_factory=list)
    confidence: float = 0.7
    metadata: dict[str, Any] = Field(default_factory=dict)


class CanonicalSkillCard(BaseModel):
    skill_id: str
    version: str = "v1"
    name: str
    mode: ModeLabel
    summary: str
    trigger: list[str] = Field(default_factory=list)
    diagnosis: list[str] = Field(default_factory=list)
    response_plan: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    negative_cases: list[str] = Field(default_factory=list)
    evidence_units: list[str] = Field(default_factory=list)
    active: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class RouterLabelRecord(BaseModel):
    sample_id: str
    query: str
    gold_mode: ModeLabel
    candidate_modes: list[ModeLabel] = Field(default_factory=list)
    route_reason: str
    evidence_unit_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.8
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalRecord(BaseModel):
    eval_id: str
    task_type: str
    input_query: str
    expected_mode: ModeLabel | None = None
    expected_traits: list[str] = Field(default_factory=list)
    expected_skills: list[str] = Field(default_factory=list)
    evidence_unit_ids: list[str] = Field(default_factory=list)
    reference_answer: str | None = None
    difficulty: str = "normal"
    metadata: dict[str, Any] = Field(default_factory=dict)

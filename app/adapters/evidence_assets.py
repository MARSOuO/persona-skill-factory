from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.domain.models import EvidenceUnit


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            rows.append(row)
    return rows


@lru_cache(maxsize=4)
def load_evidence_index(path_str: str) -> list[EvidenceUnit]:
    path = Path(path_str)
    if not path.exists():
        return []

    rows = _read_jsonl(path)
    units: list[EvidenceUnit] = []
    for row in rows:
        units.append(
            EvidenceUnit(
                evidence_id=str(row["evidence_id"]),
                unit_id=str(row.get("unit_id") or row["evidence_id"]),
                mode=str(row.get("mode") or "unknown"),
                text=str(row.get("text") or ""),
                source_file=str(row.get("source_file") or ""),
                paragraph_id=row.get("paragraph_id"),
                metadata=row.get("metadata") or {},
            )
        )
    return units


@lru_cache(maxsize=4)
def load_skill_unit_links(path_str: str) -> dict[str, list[str]]:
    path = Path(path_str)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("skill_to_unit_links must be a JSON object")
    out: dict[str, list[str]] = {}
    for skill_id, value in data.items():
        if isinstance(value, list):
            out[str(skill_id)] = [str(x) for x in value]
    return out

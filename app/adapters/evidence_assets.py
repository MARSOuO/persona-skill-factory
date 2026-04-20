from __future__ import annotations

import json
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
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(row)
    return rows


def load_evidence_index(path: str) -> list[EvidenceUnit]:
    rows = _read_jsonl(Path(path))
    units: list[EvidenceUnit] = []

    for row in rows:
        metadata = dict(row.get("metadata") or {})

        # 把 v1.1 顶层字段并入 metadata，方便 retriever 统一读取
        for key in ("token_set", "skill_hints", "text_len"):
            if key in row:
                metadata[key] = row[key]

        units.append(
            EvidenceUnit(
                evidence_id=str(row.get("evidence_id") or ""),
                unit_id=str(row.get("unit_id") or ""),
                mode=str(row.get("mode") or "unknown"),
                text=str(row.get("text") or ""),
                source_file=str(row.get("source_file") or ""),
                paragraph_id=row.get("paragraph_id"),
                metadata=metadata,
            )
        )

    return units


def load_skill_unit_links(path: str) -> dict[str, list[str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("skill_unit_links JSON must be an object")

    out: dict[str, list[str]] = {}
    for key, value in data.items():
        if isinstance(value, list):
            out[str(key)] = [str(x) for x in value if str(x).strip()]
    return out
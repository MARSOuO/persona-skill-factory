#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from app.utils.tokenize import tokenize_zh_en


def build_token_set(text: str) -> list[str]:
    return tokenize_zh_en(text)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def load_skills(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    skills = data.get("skills", data)
    if not isinstance(skills, list):
        raise ValueError("canonical_skills YAML must be a list or contain a top-level 'skills' list")
    return [x for x in skills if isinstance(x, dict)]


def infer_mode(row: dict[str, Any]) -> str:
    for key in ("mode", "router_mode", "corpus_mode", "track", "label"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    text_candidates = [
        str(row.get("source_file") or ""),
        str(row.get("filename") or ""),
        str(row.get("file_name") or ""),
        str(row.get("source") or ""),
    ]
    joined = " ".join(text_candidates).lower()
    for mode in ("teaching", "bridge", "commentary", "persona"):
        if mode in joined:
            return mode
    return "unknown"


def extract_text(row: dict[str, Any]) -> str:
    for key in ("text", "unit_text", "content", "normalized_text", "clean_text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def extract_source_file(row: dict[str, Any]) -> str:
    for key in ("source_file", "filename", "file_name", "source"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    meta = row.get("metadata") or {}
    if isinstance(meta, dict):
        for key in ("source_file", "filename", "file_name"):
            value = meta.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def extract_paragraph_id(row: dict[str, Any]) -> int | None:
    for key in ("paragraph_id", "para_id", "paragraph_index"):
        value = row.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def extract_unit_id(row: dict[str, Any], fallback: int) -> str:
    for key in ("unit_id", "id"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"unit_{fallback:06d}"


def tokenize(text: str) -> list[str]:
    # 为兼容旧逻辑保留；当前等价于 build_token_set
    return build_token_set(text)


def build_index(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index_rows: list[dict[str, Any]] = []

    for i, row in enumerate(units, start=1):
        unit_id = extract_unit_id(row, i)
        mode = infer_mode(row)
        text = extract_text(row)
        if not text:
            continue

        source_file = extract_source_file(row)
        paragraph_id = extract_paragraph_id(row)
        token_set = build_token_set(text)

        index_rows.append(
            {
                "evidence_id": f"evidence::{unit_id}",
                "unit_id": unit_id,
                "mode": mode,
                "text": text,
                "source_file": source_file,
                "paragraph_id": paragraph_id,
                "token_set": token_set,
                "text_len": len(text),
                "skill_hints": [],
                "metadata": {
                    "unit_tokens": token_set,
                },
            }
        )

    return index_rows


def build_skill_links(
    index_rows: list[dict[str, Any]],
    skills: list[dict[str, Any]],
    top_n: int,
) -> dict[str, list[str]]:
    units_by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in index_rows:
        units_by_mode[str(row.get("mode") or "unknown")].append(row)

    skill_links: dict[str, list[str]] = {}

    for skill in skills:
        skill_id = str(skill.get("skill_id") or skill.get("id") or "").strip()
        if not skill_id:
            continue

        mode = str(skill.get("mode") or "unknown")
        triggers = [str(x).lower() for x in skill.get("triggers", []) if str(x).strip()]
        anti_triggers = [str(x).lower() for x in skill.get("anti_triggers", []) if str(x).strip()]
        summary_tokens = set(tokenize(str(skill.get("summary") or "")))

        scored: list[tuple[float, str]] = []

        for unit in units_by_mode.get(mode, []):
            text = str(unit.get("text") or "").lower()
            token_set = set(unit.get("token_set", []))

            trigger_hits = sum(1 for t in triggers if t in text)
            anti_hits = sum(1 for t in anti_triggers if t in text)
            summary_hits = len(summary_tokens & token_set)

            score = 1.2 * trigger_hits + 0.08 * summary_hits - 0.4 * anti_hits
            if score > 0:
                scored.append((score, str(unit["unit_id"])))

        scored.sort(key=lambda x: x[0], reverse=True)
        skill_links[skill_id] = [unit_id for _, unit_id in scored[:top_n]]

    return skill_links


def backfill_skill_hints(index_rows: list[dict[str, Any]], links: dict[str, list[str]]) -> None:
    unit_to_skills: dict[str, list[str]] = defaultdict(list)

    for skill_id, unit_ids in links.items():
        for unit_id in unit_ids:
            unit_to_skills[str(unit_id)].append(skill_id)

    for row in index_rows:
        unit_id = str(row["unit_id"])
        row["skill_hints"] = unit_to_skills.get(unit_id, [])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--units", required=True, help="Path to all_units.jsonl or equivalent JSONL")
    parser.add_argument("--canonical-skills", required=True, help="Path to canonical_skills YAML")
    parser.add_argument("--out-index", required=True, help="Output evidence index JSONL")
    parser.add_argument("--out-links", required=True, help="Output skill_to_unit_links JSON")
    parser.add_argument("--top-n-per-skill", type=int, default=12)
    args = parser.parse_args()

    units_path = Path(args.units)
    skills_path = Path(args.canonical_skills)
    out_index = Path(args.out_index)
    out_links = Path(args.out_links)

    units = read_jsonl(units_path)
    skills = load_skills(skills_path)

    index_rows = build_index(units)
    links = build_skill_links(index_rows, skills, top_n=args.top_n_per_skill)
    backfill_skill_hints(index_rows, links)

    out_index.parent.mkdir(parents=True, exist_ok=True)
    out_links.parent.mkdir(parents=True, exist_ok=True)

    with out_index.open("w", encoding="utf-8") as f:
        for row in index_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    out_links.write_text(json.dumps(links, ensure_ascii=False, indent=2), encoding="utf-8")

    mode_counts: dict[str, int] = defaultdict(int)
    for row in index_rows:
        mode_counts[str(row.get("mode") or "unknown")] += 1

    print(
        json.dumps(
            {
                "units_in": len(units),
                "evidence_rows_out": len(index_rows),
                "mode_counts": dict(mode_counts),
                "skill_links": len(links),
                "out_index": str(out_index),
                "out_links": str(out_links),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
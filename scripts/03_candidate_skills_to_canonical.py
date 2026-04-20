from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import jsonlines
import yaml

from skill_factory.models import CandidateSkillRecord


def unique_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def shorten(items: list[str], limit: int) -> list[str]:
    return items[:limit]


def build_skill_id(mode: str, name: str) -> str:
    slug = (
        name.replace(" ", "_")
        .replace("/", "_")
        .replace("（", "_")
        .replace("）", "_")
        .replace("(", "_")
        .replace(")", "_")
        .replace("、", "_")
    )
    return f"{mode}__{slug}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="candidate_skills jsonl")
    parser.add_argument("--output", required=True, help="canonical_skills yaml")
    parser.add_argument("--min-count", type=int, default=3, help="minimum merged count to keep")
    parser.add_argument("--max-evidence", type=int, default=20, help="max evidence units per canonical skill")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    groups: dict[tuple[str, str], list[CandidateSkillRecord]] = defaultdict(list)

    with jsonlines.open(input_path, "r") as reader:
        for row in reader:
            cand = CandidateSkillRecord.model_validate(row)
            key = (cand.mode.value, cand.name)
            groups[key].append(cand)

    canonical = []

    for (mode, name), items in sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True):
        count = len(items)

        # 放宽 persona / commentary / bridge，避免小类被阉掉
        effective_min = args.min_count
        if mode in {"persona", "commentary", "bridge"}:
            effective_min = 2

        if count < effective_min:
            continue

        description = items[0].description

        trigger = unique_keep_order([x for item in items for x in item.trigger])
        diagnosis = unique_keep_order([x for item in items for x in item.diagnosis])
        response_plan = unique_keep_order([x for item in items for x in item.response_plan])
        avoid = unique_keep_order([x for item in items for x in item.avoid])
        negative_cases = unique_keep_order([x for item in items for x in item.negative_cases])

        evidence_units = unique_keep_order([x for item in items for x in item.evidence_units])
        evidence_quotes = unique_keep_order([x for item in items for x in item.evidence_quotes])

        source_files = unique_keep_order(
            [item.metadata.get("source_file") for item in items if item.metadata.get("source_file")]
        )
        filenames = unique_keep_order(
            [item.metadata.get("filename") for item in items if item.metadata.get("filename")]
        )

        avg_conf = round(sum(item.confidence for item in items) / len(items), 4)

        canonical.append(
            {
                "skill_id": build_skill_id(mode, name),
                "version": "v1",
                "name": name,
                "mode": mode,
                "summary": description,
                "trigger": shorten(trigger, 8),
                "diagnosis": shorten(diagnosis, 8),
                "response_plan": shorten(response_plan, 8),
                "avoid": shorten(avoid, 8),
                "negative_cases": shorten(negative_cases, 8),
                "evidence_units": shorten(evidence_units, args.max_evidence),
                "active": True,
                "metadata": {
                    "candidate_count": count,
                    "avg_confidence": avg_conf,
                    "source_files": shorten(source_files, 20),
                    "filenames": shorten(filenames, 20),
                    "sample_quotes": shorten(evidence_quotes, 8),
                },
            }
        )

    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            canonical,
            f,
            allow_unicode=True,
            sort_keys=False,
            width=120,
        )

    print(f"done: candidate_groups={len(groups)}, canonical_skills={len(canonical)}, output={output_path}")


if __name__ == "__main__":
    main()

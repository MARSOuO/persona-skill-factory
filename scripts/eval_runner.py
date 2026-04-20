#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import requests
import yaml


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {line_no}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no} is not a JSON object")
            rows.append(obj)
    return rows


def normalize_text(s: str) -> str:
    return " ".join(s.strip().lower().split())


def load_skill_aliases(canonical_skills_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """
    Returns:
      skill_id_to_name: canonical skill id -> canonical display name
      alias_to_skill_id: normalized alias/name/id -> canonical skill id
    """
    with canonical_skills_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    skills = data.get("skills", data)
    if not isinstance(skills, list):
        raise ValueError("canonical_skills YAML must contain a top-level 'skills' list or be a list itself")

    skill_id_to_name: dict[str, str] = {}
    alias_to_skill_id: dict[str, str] = {}

    for item in skills:
        if not isinstance(item, dict):
            continue
        skill_id = item.get("skill_id") or item.get("id")
        name = item.get("name") or item.get("title") or skill_id
        if not skill_id:
            continue

        skill_id_to_name[skill_id] = name
        alias_to_skill_id[normalize_text(skill_id)] = skill_id
        alias_to_skill_id[normalize_text(name)] = skill_id

        for alias in item.get("aliases", []) or []:
            if isinstance(alias, str) and alias.strip():
                alias_to_skill_id[normalize_text(alias)] = skill_id

        # Some schemas may include zh_name/display_name/etc.
        for extra_name_key in ("zh_name", "display_name", "label"):
            extra_name = item.get(extra_name_key)
            if isinstance(extra_name, str) and extra_name.strip():
                alias_to_skill_id[normalize_text(extra_name)] = skill_id

    return skill_id_to_name, alias_to_skill_id


def extract_query(row: dict[str, Any]) -> str:
    for key in ("query", "input_query", "prompt", "input", "user_query", "question"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def extract_task(row: dict[str, Any]) -> str:
    for key in ("task", "task_type", "eval_type", "subset", "kind"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def extract_gold_mode(row: dict[str, Any]) -> str | None:
    for key in ("gold_mode", "expected_mode", "router_label", "mode"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    target = row.get("target")
    if isinstance(target, dict):
        for key in ("mode", "gold_mode", "expected_mode"):
            value = target.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def extract_gold_skills(row: dict[str, Any]) -> list[str]:
    candidates = [
        row.get("gold_skill_ids"),
        row.get("expected_skills"),
        row.get("gold_skills"),
        row.get("skill_ids"),
    ]
    target = row.get("target")
    if isinstance(target, dict):
        candidates.extend([
            target.get("gold_skill_ids"),
            target.get("expected_skills"),
            target.get("gold_skills"),
            target.get("skill_ids"),
        ])

    for value in candidates:
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]
    return []


def normalize_skill_refs(values: list[str], alias_to_skill_id: dict[str, str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        norm = normalize_text(raw)
        skill_id = alias_to_skill_id.get(norm, raw.strip())
        if skill_id and skill_id not in seen:
            out.append(skill_id)
            seen.add(skill_id)
    return out


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def post_json(endpoint: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    resp = requests.post(endpoint, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise ValueError("Endpoint did not return a JSON object")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to normalized eval dataset JSONL")
    parser.add_argument("--endpoint", required=True, help="HTTP endpoint for /route_and_plan")
    parser.add_argument("--output", required=True, help="Where to write summary JSON")
    parser.add_argument(
        "--canonical-skills",
        default="assets/canonical_skills.v1.yaml",
        help="Path to canonical_skills YAML for mapping skill ids <-> names",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--top-k-skills", type=int, default=2)
    parser.add_argument("--top-k-evidence", type=int, default=3)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    canonical_skills_path = Path(args.canonical_skills)

    rows = load_jsonl(dataset_path)
    skill_id_to_name, alias_to_skill_id = load_skill_aliases(canonical_skills_path)

    routing_total = 0
    routing_correct = 0
    confusion: dict[str, Counter] = defaultdict(Counter)

    skill_total = 0
    skill_hit = 0
    skill_jaccards: list[float] = []

    task_counts: Counter = Counter()
    errors: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        query = extract_query(row)
        task = extract_task(row)
        gold_mode = extract_gold_mode(row)
        gold_skills_raw = extract_gold_skills(row)
        gold_skills = normalize_skill_refs(gold_skills_raw, alias_to_skill_id)

        task_counts[task] += 1

        payload = {
            "query": query,
            "top_k_skills": args.top_k_skills,
            "top_k_evidence": args.top_k_evidence,
        }
        pred = post_json(args.endpoint, payload, timeout=args.timeout)

        predicted_mode = (
            ((pred.get("predicted_mode") or {}).get("label"))
            if isinstance(pred.get("predicted_mode"), dict)
            else None
        )
        predicted_skills_raw = []
        for item in pred.get("selected_skills", []) or []:
            if isinstance(item, dict):
                skill_id = item.get("skill_id") or item.get("id") or item.get("name")
                if skill_id:
                    predicted_skills_raw.append(str(skill_id))

        predicted_skills = normalize_skill_refs(predicted_skills_raw, alias_to_skill_id)

        row_error: dict[str, Any] = {
            "row_index": idx,
            "task": task,
            "query": query,
            "predicted_mode": predicted_mode,
            "predicted_skills": predicted_skills,
        }

        if gold_mode:
            routing_total += 1
            confusion[gold_mode][predicted_mode or "null"] += 1
            if predicted_mode == gold_mode:
                routing_correct += 1
            else:
                row_error["gold_mode"] = gold_mode
                row_error["mode_error"] = True

        if task == "skill_eval" or gold_skills:
            skill_total += 1
            pred_set = set(predicted_skills)
            gold_set = set(gold_skills)

            jac = round(jaccard(pred_set, gold_set), 4)
            hit = len(pred_set & gold_set) > 0

            skill_jaccards.append(jac)
            if hit:
                skill_hit += 1

            if not hit or jac < 1.0:
                row_error["gold_skills"] = gold_skills
                row_error["gold_skills_display"] = [skill_id_to_name.get(s, s) for s in gold_skills]
                row_error["predicted_skills_display"] = [skill_id_to_name.get(s, s) for s in predicted_skills]
                row_error["skill_jaccard"] = jac
                row_error["skill_error"] = True

        if "mode_error" in row_error or "skill_error" in row_error:
            errors.append(row_error)

    ordered_labels = sorted(set(confusion.keys()) | {k for c in confusion.values() for k in c.keys()})
    confusion_matrix: dict[str, dict[str, int]] = {}
    for gold in ordered_labels:
        confusion_matrix[gold] = {pred: confusion[gold][pred] for pred in ordered_labels}

    summary = {
        "records": len(rows),
        "task_counts": dict(task_counts),
        "routing": {
            "total": routing_total,
            "accuracy": round(routing_correct / routing_total, 4) if routing_total else 0.0,
            "correct": routing_correct,
            "confusion_matrix": confusion_matrix,
        },
        "skills": {
            "total": skill_total,
            "top_k_hit_rate": round(skill_hit / skill_total, 4) if skill_total else 0.0,
            "mean_jaccard": round(sum(skill_jaccards) / len(skill_jaccards), 4) if skill_jaccards else 0.0,
        },
        "errors": {
            "count": len(errors),
            "sample": errors[:25],
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved summary to: {output_path}")


if __name__ == "__main__":
    main()

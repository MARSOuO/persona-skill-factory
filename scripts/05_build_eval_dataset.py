from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import jsonlines
import yaml

from skill_factory.models import EvalRecord, ModeLabel, RouterLabelRecord


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def make_eval_id(prefix: str, split: str, query: str, salt: str) -> str:
    h = hashlib.md5(f"{prefix}|{split}|{query}|{salt}".encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{split}_{h}"


def load_router_labels(path: Path) -> list[RouterLabelRecord]:
    records = []
    with jsonlines.open(path, "r") as reader:
        for row in reader:
            records.append(RouterLabelRecord.model_validate(row))
    return records


def load_canonical_skills(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or []


def build_canonical_indexes(canonical_skills: list[dict]) -> tuple[dict[str, dict], dict[str, set[str]]]:
    by_name = {}
    names_by_mode: dict[str, set[str]] = defaultdict(set)

    for skill in canonical_skills:
        by_name[skill["name"]] = skill
        names_by_mode[skill["mode"]].add(skill["name"])

    return by_name, names_by_mode


def pick_existing(name: str, names_by_mode: dict[str, set[str]], mode: str) -> list[str]:
    if name in names_by_mode.get(mode, set()):
        return [name]
    return []


def infer_skill_names(query: str, gold_mode: ModeLabel, names_by_mode: dict[str, set[str]]) -> list[str]:
    q = normalize_text(query)
    mode = gold_mode.value

    if mode == "teaching":
        if "分步骤" in q:
            return pick_existing("分步骤推进讲解", names_by_mode, mode) or ["教学解释推进"]
        if "先判断" in q or "选方法" in q:
            return pick_existing("先判断再选方法", names_by_mode, mode) or ["教学解释推进"]
        if "举个例子" in q or "再解释" in q:
            return pick_existing("用类比或例子导入知识点", names_by_mode, mode) or ["教学解释推进"]
        if "总结" in q:
            return pick_existing("讲解后做归纳收束", names_by_mode, mode) or ["教学解释推进"]
        return pick_existing("教学解释推进", names_by_mode, mode) or []

    if mode == "bridge":
        return pick_existing("轻松过渡式讲解", names_by_mode, mode) or []

    if mode == "commentary":
        if "本质" in q or "问题到底在哪" in q:
            return pick_existing("指出问题本质", names_by_mode, mode) or ["评论性判断输出"]
        if "表态" in q or "怎么看" in q:
            # 当前 canonical 里一般两条都在，优先更强的“明确表态”
            chosen = pick_existing("明确表态式判断输出", names_by_mode, mode)
            if chosen:
                return chosen
            return pick_existing("评论性判断输出", names_by_mode, mode) or []
        return pick_existing("评论性判断输出", names_by_mode, mode) or []

    if mode == "persona":
        return pick_existing("人设化互动表达", names_by_mode, mode) or []

    return []


def build_reference_answer(skill_names: list[str], canonical_by_name: dict[str, dict]) -> str | None:
    if not skill_names:
        return None

    parts = []
    for name in skill_names:
        skill = canonical_by_name.get(name)
        if not skill:
            continue

        summary = skill.get("summary", "")
        response_plan = skill.get("response_plan", [])[:3]
        if response_plan:
            parts.append(f"{name}：{summary} 关键动作：{' / '.join(response_plan)}")
        else:
            parts.append(f"{name}：{summary}")

    if not parts:
        return None
    return " ".join(parts)


def build_routing_eval(
    records: list[RouterLabelRecord],
    split: str,
) -> list[EvalRecord]:
    out: list[EvalRecord] = []

    for rec in records:
        out.append(
            EvalRecord(
                eval_id=make_eval_id("routing", split, rec.query, rec.sample_id),
                task_type="routing_eval",
                input_query=rec.query,
                expected_mode=rec.gold_mode,
                expected_traits=[],
                expected_skills=[],
                evidence_unit_ids=rec.evidence_unit_ids,
                reference_answer=None,
                difficulty="normal",
                metadata={
                    "split": split,
                    "source_file": rec.metadata.get("source_file"),
                    "filename": rec.metadata.get("filename"),
                    "paragraph_id": rec.metadata.get("paragraph_id"),
                    "route_reason": rec.route_reason,
                    "candidate_modes": [m.value for m in rec.candidate_modes],
                    "text_preview": rec.metadata.get("text_preview"),
                },
            )
        )

    return out


def build_skill_eval(
    records: list[RouterLabelRecord],
    split: str,
    canonical_by_name: dict[str, dict],
    names_by_mode: dict[str, set[str]],
) -> list[EvalRecord]:
    out: list[EvalRecord] = []

    for rec in records:
        skill_names = infer_skill_names(rec.query, rec.gold_mode, names_by_mode)
        ref_answer = build_reference_answer(skill_names, canonical_by_name)

        out.append(
            EvalRecord(
                eval_id=make_eval_id("skill", split, rec.query, rec.sample_id),
                task_type="skill_eval",
                input_query=rec.query,
                expected_mode=rec.gold_mode,
                expected_traits=skill_names[:],
                expected_skills=skill_names,
                evidence_unit_ids=rec.evidence_unit_ids,
                reference_answer=ref_answer,
                difficulty="normal",
                metadata={
                    "split": split,
                    "source_file": rec.metadata.get("source_file"),
                    "filename": rec.metadata.get("filename"),
                    "paragraph_id": rec.metadata.get("paragraph_id"),
                    "route_reason": rec.route_reason,
                    "text_preview": rec.metadata.get("text_preview"),
                },
            )
        )

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-dev", required=True, help="strict dev router labels jsonl")
    parser.add_argument("--router-test", required=True, help="strict test router labels jsonl")
    parser.add_argument("--canonical", required=True, help="canonical skills yaml")
    parser.add_argument("--output", required=True, help="combined eval_dataset jsonl")
    parser.add_argument("--report", required=True, help="eval dataset report json")
    args = parser.parse_args()

    dev_path = Path(args.router_dev)
    test_path = Path(args.router_test)
    canonical_path = Path(args.canonical)
    output_path = Path(args.output)
    report_path = Path(args.report)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    dev_records = load_router_labels(dev_path)
    test_records = load_router_labels(test_path)
    canonical_skills = load_canonical_skills(canonical_path)

    canonical_by_name, names_by_mode = build_canonical_indexes(canonical_skills)

    routing_dev = build_routing_eval(dev_records, "dev")
    routing_test = build_routing_eval(test_records, "test")
    skill_dev = build_skill_eval(dev_records, "dev", canonical_by_name, names_by_mode)
    skill_test = build_skill_eval(test_records, "test", canonical_by_name, names_by_mode)

    all_records = routing_dev + routing_test + skill_dev + skill_test

    with jsonlines.open(output_path, "w") as writer:
        for rec in all_records:
            writer.write(rec.model_dump(mode="json"))

    # report
    task_counter = Counter()
    split_counter = Counter()
    task_split_counter = defaultdict(Counter)
    mode_counter = Counter()
    skill_name_counter = Counter()

    for rec in all_records:
        task_counter[rec.task_type] += 1
        split = rec.metadata.get("split", "unknown")
        split_counter[split] += 1
        task_split_counter[rec.task_type][split] += 1
        if rec.expected_mode is not None:
            mode_counter[rec.expected_mode.value] += 1
        for s in rec.expected_skills:
            skill_name_counter[s] += 1

    report = {
        "total_eval_records": len(all_records),
        "task_counts": dict(task_counter),
        "split_counts": dict(split_counter),
        "task_split_counts": {k: dict(v) for k, v in task_split_counter.items()},
        "expected_mode_counts": dict(mode_counter),
        "top_expected_skills": skill_name_counter.most_common(20),
        "router_dev_count": len(dev_records),
        "router_test_count": len(test_records),
        "canonical_skill_count": len(canonical_skills),
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(
        f"done: routing_dev={len(routing_dev)}, routing_test={len(routing_test)}, "
        f"skill_dev={len(skill_dev)}, skill_test={len(skill_test)}, total={len(all_records)}"
    )


if __name__ == "__main__":
    main()

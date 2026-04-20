from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import jsonlines

from skill_factory.models import ModeLabel, RouterLabelRecord


MODE_ORDER = [
    ModeLabel.teaching,
    ModeLabel.bridge,
    ModeLabel.commentary,
    ModeLabel.persona,
]


def normalize_query(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def record_to_plain(rec: RouterLabelRecord) -> dict:
    return rec.model_dump(mode="json")


def score_record(rec: RouterLabelRecord) -> tuple:
    preview = rec.metadata.get("text_preview", "") or ""
    style_n = len(rec.metadata.get("style_signals", []) or [])
    teach_n = len(rec.metadata.get("teaching_signals", []) or [])
    reason_len = len(rec.route_reason)
    query_len = len(rec.query)
    preview_len = len(preview)

    return (
        -(style_n + teach_n),
        -reason_len,
        -preview_len,
        -query_len,
        rec.sample_id,
    )


def exact_dedup(records: list[RouterLabelRecord]) -> list[RouterLabelRecord]:
    seen = set()
    out = []
    for rec in records:
        source_file = rec.metadata.get("source_file", "")
        preview = rec.metadata.get("text_preview", "")
        key = (
            rec.gold_mode.value,
            normalize_query(rec.query),
            source_file,
            preview,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


def cap_per_query(records: list[RouterLabelRecord], per_query_cap: int) -> list[RouterLabelRecord]:
    buckets: dict[tuple[str, str], list[RouterLabelRecord]] = defaultdict(list)
    for rec in records:
        key = (rec.gold_mode.value, normalize_query(rec.query))
        buckets[key].append(rec)

    capped = []
    for _, items in buckets.items():
        items = sorted(items, key=score_record)
        capped.extend(items[:per_query_cap])
    return capped


def group_by_source(records: list[RouterLabelRecord]) -> dict[str, list[RouterLabelRecord]]:
    groups: dict[str, list[RouterLabelRecord]] = defaultdict(list)
    for rec in records:
        source = rec.metadata.get("source_file", "__unknown__")
        groups[source].append(rec)
    return groups


def source_mode_counter(records: list[RouterLabelRecord]) -> Counter:
    c = Counter()
    for rec in records:
        c[rec.gold_mode.value] += 1
    return c


def choose_eval_sources(
    source_groups: dict[str, list[RouterLabelRecord]],
    eval_per_mode: int,
    excluded_sources: set[str] | None = None,
) -> tuple[set[str], Counter]:
    if excluded_sources is None:
        excluded_sources = set()

    selected: set[str] = set()
    current = Counter()

    available = {
        s: source_mode_counter(items)
        for s, items in source_groups.items()
        if s not in excluded_sources
    }

    target = {m.value: eval_per_mode for m in MODE_ORDER}

    while True:
        deficits = {m: max(0, target[m] - current[m]) for m in target}
        if all(v == 0 for v in deficits.values()):
            break

        best_source = None
        best_score = None

        for source, mode_counts in available.items():
            if source in selected:
                continue

            gain = sum(min(deficits[m], mode_counts[m]) for m in target)
            if gain == 0:
                continue

            overshoot = sum(max(0, mode_counts[m] - deficits[m]) for m in target)
            total = sum(mode_counts.values())
            dominant = max(mode_counts.values()) if mode_counts else 0

            # 优先补齐缺口，其次少超量，其次小源文件，最后更单一的源
            score = (-gain, overshoot, total, -dominant, source)

            if best_score is None or score < best_score:
                best_score = score
                best_source = source

        if best_source is None:
            break

        selected.add(best_source)
        for rec in source_groups[best_source]:
            current[rec.gold_mode.value] += 1

    return selected, current


def balanced_take(records: list[RouterLabelRecord], per_mode: int) -> list[RouterLabelRecord]:
    by_mode: dict[str, list[RouterLabelRecord]] = defaultdict(list)
    for rec in records:
        by_mode[rec.gold_mode.value].append(rec)

    out = []
    for mode in [m.value for m in MODE_ORDER]:
        items = sorted(by_mode.get(mode, []), key=score_record)
        out.extend(items[:per_mode])
    return out


def predict_mode(query: str) -> str:
    q = normalize_query(query)

    if any(k in q for k in ["互动", "有人味", "陪我聊聊", "别太正经", "平时那种口气"]):
        return "persona"
    if any(k in q for k in ["怎么看", "表态", "判断", "本质", "问题", "别只复述"]):
        return "commentary"
    if any(k in q for k in ["温和", "带我入门", "轻松", "别太硬核", "慢慢", "先温和地"]):
        return "bridge"
    return "teaching"


def eval_split(records: list[RouterLabelRecord]) -> dict:
    gold_counter = Counter()
    pred_counter = Counter()
    confusion: dict[str, Counter] = defaultdict(Counter)
    correct = 0

    for rec in records:
        gold = rec.gold_mode.value
        pred = predict_mode(rec.query)
        gold_counter[gold] += 1
        pred_counter[pred] += 1
        confusion[gold][pred] += 1
        if gold == pred:
            correct += 1

    total = len(records)
    acc = round(correct / total, 4) if total else 0.0
    return {
        "count": total,
        "accuracy": acc,
        "gold": dict(gold_counter),
        "pred": dict(pred_counter),
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }


def save_jsonl(path: Path, records: list[RouterLabelRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, "w") as writer:
        for rec in records:
            writer.write(record_to_plain(rec))


def overlap_stats(a: list[RouterLabelRecord], b: list[RouterLabelRecord]) -> dict:
    a_sources = {x.metadata.get("source_file", "") for x in a}
    b_sources = {x.metadata.get("source_file", "") for x in b}
    a_queries = {normalize_query(x.query) for x in a}
    b_queries = {normalize_query(x.query) for x in b}

    return {
        "shared_sources": len(a_sources & b_sources),
        "shared_queries": len(a_queries & b_queries),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="router_labels.v1.jsonl")
    parser.add_argument("--output-dir", required=True, help="output dir")
    parser.add_argument("--per-query-cap", type=int, default=80)
    parser.add_argument("--eval-per-mode", type=int, default=20)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_records = []
    with jsonlines.open(input_path, "r") as reader:
        for row in reader:
            raw_records.append(RouterLabelRecord.model_validate(row))

    raw_mode = Counter(x.gold_mode.value for x in raw_records)

    dedup_records = exact_dedup(raw_records)
    dedup_mode = Counter(x.gold_mode.value for x in dedup_records)

    capped_records = cap_per_query(dedup_records, args.per_query_cap)
    capped_mode = Counter(x.gold_mode.value for x in capped_records)

    source_groups = group_by_source(capped_records)

    dev_sources, dev_source_mode = choose_eval_sources(
        source_groups=source_groups,
        eval_per_mode=args.eval_per_mode,
        excluded_sources=set(),
    )

    test_sources, test_source_mode = choose_eval_sources(
        source_groups=source_groups,
        eval_per_mode=args.eval_per_mode,
        excluded_sources=dev_sources,
    )

    dev_pool = []
    test_pool = []
    train_pool = []

    for source, items in source_groups.items():
        if source in dev_sources:
            dev_pool.extend(items)
        elif source in test_sources:
            test_pool.extend(items)
        else:
            train_pool.extend(items)

    # dev/test 各自再平衡采样；train 保持较大
    dev_records = balanced_take(dev_pool, args.eval_per_mode)
    test_records = balanced_take(test_pool, args.eval_per_mode)
    train_records = train_pool

    train_mode = Counter(x.gold_mode.value for x in train_records)
    dev_mode = Counter(x.gold_mode.value for x in dev_records)
    test_mode = Counter(x.gold_mode.value for x in test_records)

    strict_train_path = out_dir / "router_labels.v1.strict.train.jsonl"
    strict_dev_path = out_dir / "router_labels.v1.strict.dev.jsonl"
    strict_test_path = out_dir / "router_labels.v1.strict.test.jsonl"
    strict_report_path = out_dir / "router_labels.v1.strict.report.json"

    save_jsonl(strict_train_path, train_records)
    save_jsonl(strict_dev_path, dev_records)
    save_jsonl(strict_test_path, test_records)

    report = {
        "raw_count": len(raw_records),
        "raw_mode": dict(raw_mode),
        "dedup_count": len(dedup_records),
        "dedup_mode": dict(dedup_mode),
        "capped_count": len(capped_records),
        "capped_mode": dict(capped_mode),
        "per_query_cap": args.per_query_cap,
        "eval_per_mode": args.eval_per_mode,
        "dev_source_mode_before_sample": dict(dev_source_mode),
        "test_source_mode_before_sample": dict(test_source_mode),
        "splits": {
            "train_count": len(train_records),
            "train_mode": dict(train_mode),
            "dev_count": len(dev_records),
            "dev_mode": dict(dev_mode),
            "test_count": len(test_records),
            "test_mode": dict(test_mode),
        },
        "overlap": {
            "train_dev": overlap_stats(train_records, dev_records),
            "train_test": overlap_stats(train_records, test_records),
            "dev_test": overlap_stats(dev_records, test_records),
        },
        "baseline_eval": {
            "dev": eval_split(dev_records),
            "test": eval_split(test_records),
        },
        "source_counts": {
            "train_sources": len({x.metadata.get("source_file", "") for x in train_records}),
            "dev_sources": len({x.metadata.get("source_file", "") for x in dev_records}),
            "test_sources": len({x.metadata.get("source_file", "") for x in test_records}),
        },
    }

    with strict_report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(
        f"done: raw={len(raw_records)}, dedup={len(dedup_records)}, capped={len(capped_records)}, "
        f"train={len(train_records)}, dev={len(dev_records)}, test={len(test_records)}, "
        f"eval_per_mode={args.eval_per_mode}"
    )


if __name__ == "__main__":
    main()

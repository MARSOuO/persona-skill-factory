from __future__ import annotations

import argparse
import json
import math
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

TEACHING_HINTS = ["讲", "步骤", "判断", "方法", "知识点", "总结", "例子", "解释"]
BRIDGE_HINTS = ["温和", "带我入门", "轻松", "别太硬核", "慢慢", "先带我"]
COMMENTARY_HINTS = ["怎么看", "表态", "判断", "本质", "问题", "别只复述"]
PERSONA_HINTS = ["互动", "有人味", "陪我聊聊", "别太正经", "平时那种口气"]


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

    # 越靠前越优
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
    for key, items in buckets.items():
        items = sorted(items, key=score_record)
        capped.extend(items[:per_query_cap])
    return capped


def balanced_sample(records: list[RouterLabelRecord], target_per_mode: int) -> list[RouterLabelRecord]:
    by_mode: dict[str, list[RouterLabelRecord]] = defaultdict(list)
    for rec in records:
        by_mode[rec.gold_mode.value].append(rec)

    final = []
    for mode in [m.value for m in MODE_ORDER]:
        mode_records = by_mode[mode]
        query_buckets: dict[str, list[RouterLabelRecord]] = defaultdict(list)
        for rec in mode_records:
            query_buckets[normalize_query(rec.query)].append(rec)

        for q in query_buckets:
            query_buckets[q] = sorted(query_buckets[q], key=score_record)

        queries = sorted(query_buckets.keys(), key=lambda q: (len(query_buckets[q]), q))
        picked = []
        used_ids = set()

        while len(picked) < target_per_mode:
            progressed = False
            for q in queries:
                bucket = query_buckets[q]
                while bucket and bucket[0].sample_id in used_ids:
                    bucket.pop(0)
                if not bucket:
                    continue
                rec = bucket.pop(0)
                if rec.sample_id in used_ids:
                    continue
                picked.append(rec)
                used_ids.add(rec.sample_id)
                progressed = True
                if len(picked) >= target_per_mode:
                    break
            if not progressed:
                break

        final.extend(picked[:target_per_mode])

    return final


def split_by_source(
    records: list[RouterLabelRecord],
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
) -> tuple[list[RouterLabelRecord], list[RouterLabelRecord], list[RouterLabelRecord]]:
    by_mode: dict[str, list[RouterLabelRecord]] = defaultdict(list)
    for rec in records:
        by_mode[rec.gold_mode.value].append(rec)

    train, dev, test = [], [], []

    for mode in [m.value for m in MODE_ORDER]:
        mode_records = by_mode[mode]
        groups: dict[str, list[RouterLabelRecord]] = defaultdict(list)
        for rec in mode_records:
            source_file = rec.metadata.get("source_file", "__unknown__")
            groups[source_file].append(rec)

        total = len(mode_records)
        target_train = int(round(total * train_ratio))
        target_dev = int(round(total * dev_ratio))
        target_test = total - target_train - target_dev

        buckets = [("train", train, target_train), ("dev", dev, target_dev), ("test", test, target_test)]
        current = {"train": 0, "dev": 0, "test": 0}

        # 大文件先分配，避免后面塞不进去
        ordered_groups = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)

        for _, items in ordered_groups:
            best_name = None
            best_score = None
            for split_name, _, target in buckets:
                # 目标不足优先；超目标越少越好
                after = current[split_name] + len(items)
                overflow = max(0, after - target)
                underfill = max(0, target - current[split_name])
                score = (overflow, -underfill, current[split_name])
                if best_score is None or score < best_score:
                    best_score = score
                    best_name = split_name

            if best_name == "train":
                train.extend(items)
            elif best_name == "dev":
                dev.extend(items)
            else:
                test.extend(items)

            current[best_name] += len(items)

    return train, dev, test


def predict_mode(query: str) -> str:
    q = normalize_query(query)
    if any(k in q for k in PERSONA_HINTS):
        return "persona"
    if any(k in q for k in COMMENTARY_HINTS):
        return "commentary"
    if any(k in q for k in BRIDGE_HINTS):
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


def overlap_stats(a: list[RouterLabelRecord], b: list[RouterLabelRecord]) -> dict:
    a_sources = {x.metadata.get("source_file", "") for x in a}
    b_sources = {x.metadata.get("source_file", "") for x in b}
    a_queries = {normalize_query(x.query) for x in a}
    b_queries = {normalize_query(x.query) for x in b}

    return {
        "shared_sources": len(a_sources & b_sources),
        "shared_queries": len(a_queries & b_queries),
    }


def save_jsonl(path: Path, records: list[RouterLabelRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, "w") as writer:
        for rec in records:
            writer.write(record_to_plain(rec))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="router_labels.v1.jsonl")
    parser.add_argument("--output-dir", required=True, help="output directory")
    parser.add_argument("--per-query-cap", type=int, default=80, help="max rows kept per (mode, query)")
    parser.add_argument("--target-per-mode", type=int, default=0, help="0 means auto=min available after cap")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--dev-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()

    if not math.isclose(args.train_ratio + args.dev_ratio + args.test_ratio, 1.0, rel_tol=1e-6):
        raise ValueError("train/dev/test ratios must sum to 1.0")

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

    if args.target_per_mode > 0:
        target_per_mode = args.target_per_mode
    else:
        available = [capped_mode.get(m.value, 0) for m in MODE_ORDER]
        target_per_mode = min(available)

    balanced_records = balanced_sample(capped_records, target_per_mode)
    balanced_mode = Counter(x.gold_mode.value for x in balanced_records)

    train, dev, test = split_by_source(
        balanced_records,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
    )

    train_mode = Counter(x.gold_mode.value for x in train)
    dev_mode = Counter(x.gold_mode.value for x in dev)
    test_mode = Counter(x.gold_mode.value for x in test)

    balanced_path = out_dir / "router_labels.v1.balanced.jsonl"
    train_path = out_dir / "router_labels.v1.balanced.train.jsonl"
    dev_path = out_dir / "router_labels.v1.balanced.dev.jsonl"
    test_path = out_dir / "router_labels.v1.balanced.test.jsonl"
    report_path = out_dir / "router_labels.v1.balanced.report.json"

    save_jsonl(balanced_path, balanced_records)
    save_jsonl(train_path, train)
    save_jsonl(dev_path, dev)
    save_jsonl(test_path, test)

    report = {
        "raw_count": len(raw_records),
        "raw_mode": dict(raw_mode),
        "dedup_count": len(dedup_records),
        "dedup_mode": dict(dedup_mode),
        "capped_count": len(capped_records),
        "capped_mode": dict(capped_mode),
        "per_query_cap": args.per_query_cap,
        "target_per_mode": target_per_mode,
        "balanced_count": len(balanced_records),
        "balanced_mode": dict(balanced_mode),
        "splits": {
            "train_count": len(train),
            "train_mode": dict(train_mode),
            "dev_count": len(dev),
            "dev_mode": dict(dev_mode),
            "test_count": len(test),
            "test_mode": dict(test_mode),
        },
        "overlap": {
            "train_dev": overlap_stats(train, dev),
            "train_test": overlap_stats(train, test),
            "dev_test": overlap_stats(dev, test),
        },
        "baseline_eval": {
            "dev": eval_split(dev),
            "test": eval_split(test),
        },
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(
        f"done: raw={len(raw_records)}, dedup={len(dedup_records)}, capped={len(capped_records)}, "
        f"balanced={len(balanced_records)}, train={len(train)}, dev={len(dev)}, test={len(test)}, "
        f"target_per_mode={target_per_mode}"
    )


if __name__ == "__main__":
    main()

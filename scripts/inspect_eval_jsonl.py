#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from collections import Counter


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"JSON parse error at line {i}: {e}")
    return rows


def flatten_keys(obj, prefix=""):
    keys = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            keys.append(new_prefix)
            keys.extend(flatten_keys(v, new_prefix))
    elif isinstance(obj, list):
        if obj and isinstance(obj[0], dict):
            keys.extend(flatten_keys(obj[0], f"{prefix}[0]"))
    return keys


def get_by_path(obj, path, default=None):
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--preview", type=int, default=3)
    args = parser.parse_args()

    rows = load_jsonl(args.input)

    print(f"[OK] file: {args.input}")
    print(f"[OK] total rows: {len(rows)}")

    if not rows:
        return

    first = rows[0]

    print("\n=== TOP LEVEL KEYS ===")
    print(sorted(first.keys()))

    print("\n=== FLATTENED KEYS (first sample) ===")
    all_keys = sorted(set(flatten_keys(first)))
    for k in all_keys[:200]:
        print(k)

    candidate_text_fields = [
        "answer_text",
        "answer",
        "response",
        "response_text",
        "draft",
        "reference_answer",
        "metadata.text_preview",
        "output",
        "output.text",
        "response.answer",
        "response.commentary",
        "result.answer",
        "result.text",
    ]

    task_counter = Counter()
    mode_counter = Counter()

    print("\n=== PREVIEW ===")
    for i, row in enumerate(rows[:args.preview]):
        task_counter[str(row.get("task_type"))] += 1
        mode_counter[str(row.get("expected_mode") or row.get("gold_mode") or row.get("mode") or "")] += 1

        print(f"\n--- sample {i} ---")
        print("eval_id:", row.get("eval_id"))
        print("task_type:", row.get("task_type"))
        print("expected_mode:", row.get("expected_mode"))
        print("gold_mode:", row.get("gold_mode"))
        print("mode:", row.get("mode"))
        print("query:", row.get("input_query") or row.get("query"))

        print("candidate text fields:")
        for field in candidate_text_fields:
            val = get_by_path(row, field)
            if val is not None:
                s = str(val).replace("\n", " ")
                print(f"  {field}: {s[:200]}")

    print("\n=== TASK TYPE COUNTS (preview only) ===")
    print(task_counter)

    print("\n=== MODE COUNTS (preview only) ===")
    print(mode_counter)

    print("\n=== NON-EMPTY TEXT FIELD COUNTS (full file) ===")
    for field in candidate_text_fields:
        cnt = 0
        for row in rows:
            val = get_by_path(row, field)
            if val not in (None, "", [], {}):
                cnt += 1
        print(f"{field}: {cnt}")


if __name__ == "__main__":
    main()

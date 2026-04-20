#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODE_SIGNAL_MAP = {
    "teaching": {
        "expected_signals": [
            "\u5148",      # 先
            "\u6b65\u9aa4",  # 步骤
            "\u6700\u540e",  # 最后
            "\u7b2c\u4e00",  # 第一
            "\u7b2c\u4e8c",  # 第二
        ],
        "forbidden_signals": [
            "\u4e00\u65b9\u9762",      # 一方面
            "\u53e6\u4e00\u65b9\u9762",  # 另一方面
            "\u90fd\u5bf9",            # 都对
            "\u90fd\u53ef\u4ee5",      # 都可以
        ],
    },
    "bridge": {
        "expected_signals": [
            "\u5148\u522b\u6025",  # 先别急
            "\u987a\u7740",        # 顺着
            "\u6162\u6162",        # 慢慢
            "\u5e26\u4f60",        # 带你
            "\u5148\u628a\u5b83",  # 先把它
        ],
        "forbidden_signals": [
            "\u4e00\u65b9\u9762",
            "\u53e6\u4e00\u65b9\u9762",
            "\u90fd\u5bf9",
            "\u90fd\u53ef\u4ee5",
        ],
    },
    "commentary": {
        "expected_signals": [
            "\u5224\u65ad",          # 判断
            "\u672c\u8d28",          # 本质
            "\u95ee\u9898\u5728\u4e8e",  # 问题在于
            "\u6240\u4ee5",          # 所以
            "\u6211\u5148\u76f4\u63a5",  # 我先直接
        ],
        "forbidden_signals": [
            "\u4e00\u65b9\u9762",
            "\u53e6\u4e00\u65b9\u9762",
            "\u90fd\u5bf9",
            "\u90fd\u53ef\u4ee5",
        ],
    },
    "persona": {
        "expected_signals": [
            "\u4f60",          # 你
            "\u6211",          # 我
            "\u54b1",          # 咱
            "\u8ddf\u6211",    # 跟我
            "\u6211\u4eec",    # 我们
        ],
        "forbidden_signals": [
            "\u5b9a\u4e49",    # 定义
            "\u7b2c\u4e00",    # 第一
            "\u7b2c\u4e8c",    # 第二
            "\u7b2c\u4e09",    # 第三
        ],
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        query = row.get("input_query") or row.get("query") or ""
        expected_mode = row.get("expected_mode") or ""
        split = (row.get("metadata") or {}).get("split", "unknown")
        difficulty = row.get("difficulty", "normal")
        task_type = row.get("task_type", "unknown")
        expected_skills = row.get("expected_skills", [])

        if not query or not expected_mode:
            continue

        signals = MODE_SIGNAL_MAP.get(expected_mode, {})
        out.append(
            {
                "eval_id": f"answer_eval_{idx:04d}",
                "query": query,
                "expected_mode": expected_mode,
                "expected_skill_ids": expected_skills,
                "expected_signals": signals.get("expected_signals", []),
                "forbidden_signals": signals.get("forbidden_signals", []),
                "split": split,
                "difficulty": difficulty,
                "task_type": task_type,
                "metadata": row.get("metadata", {}),
            }
        )

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to eval_dataset.v1.jsonl")
    parser.add_argument("--output", required=True, help="Path to answer_eval.v1.jsonl")
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input))
    out = build_records(rows)
    write_jsonl(Path(args.output), out)

    summary = {
        "input_rows": len(rows),
        "output_rows": len(out),
        "output_path": args.output,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any


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


def post_json(url: str, payload: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def contains_any(text: str, signals: list[str]) -> bool:
    return any(sig in text for sig in signals)


def commentary_forbidden_hit(text: str) -> bool:
    """
    v1 evaluator:
    only block strong, explicit fence-sitting phrasing.
    Do NOT block weak literal mentions like "都对",
    because many valid commentary answers mention them only to reject them.
    """
    text = text.strip()

    strong_bad_patterns = [
        r"\u4e00\u65b9\u9762.{0,20}\u53e6\u4e00\u65b9\u9762",  # 一方面...另一方面
        r"\u4e24\u8fb9\u90fd\u8bf4",  # 两边都说
        r"\u90fd\u53ef\u4ee5",  # 都可以
        r"\u90fd\u6ca1\u95ee\u9898",  # 都没问题
        r"\u90fd\u5bf9\u7684",  # 都对的
        r"\u90fd\u5bf9\u554a",  # 都对啊
        r"\u90fd\u6210\u7acb",  # 都成立
    ]

    for pat in strong_bad_patterns:
        if re.search(pat, text):
            return True

    return False


def evaluate_one(row: dict[str, Any], endpoint: str, allow_llm_answer: bool) -> dict[str, Any]:
    payload = {
        "query": row["query"],
        "top_k_skills": 2,
        "top_k_evidence": 5,
        "allow_llm_answer": allow_llm_answer,
        "return_plan": False,
    }

    resp = post_json(endpoint, payload)
    answer_text = resp.get("answer_text", "")
    answer_mode = resp.get("answer_mode", "")
    generation_mode = resp.get("generation_mode", "")
    grounding = resp.get("grounding_summary", {}) or {}

    expected_mode = row["expected_mode"]
    expected_signals = row.get("expected_signals", [])
    forbidden_signals = row.get("forbidden_signals", [])

    mode_ok = answer_mode == expected_mode
    grounding_ok = not grounding.get("has_empty_grounding", True)
    drift_ok = not grounding.get("has_mode_drift", True)
    signal_hit = contains_any(answer_text, expected_signals) if expected_signals else True

    if expected_mode == "commentary":
        forbidden_hit = commentary_forbidden_hit(answer_text)
    else:
        forbidden_hit = contains_any(answer_text, forbidden_signals) if forbidden_signals else False

    overall_pass = mode_ok and grounding_ok and drift_ok and signal_hit and (not forbidden_hit)

    return {
        "eval_id": row["eval_id"],
        "query": row["query"],
        "expected_mode": expected_mode,
        "answer_mode": answer_mode,
        "generation_mode": generation_mode,
        "mode_ok": mode_ok,
        "grounding_ok": grounding_ok,
        "drift_ok": drift_ok,
        "signal_hit": signal_hit,
        "forbidden_hit": forbidden_hit,
        "overall_pass": overall_pass,
        "used_evidence_ids": resp.get("used_evidence_ids", []),
        "answer_text": answer_text,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_mode[r["expected_mode"]].append(r)

    def metric(rows: list[dict[str, Any]], key: str) -> float:
        if not rows:
            return 0.0
        return round(sum(1 for r in rows if r[key]) / len(rows), 4)

    summary = {
        "records": len(results),
        "overall": {
            "mode_accuracy": metric(results, "mode_ok"),
            "grounding_ok_rate": metric(results, "grounding_ok"),
            "drift_ok_rate": metric(results, "drift_ok"),
            "signal_hit_rate": metric(results, "signal_hit"),
            "overall_pass_rate": metric(results, "overall_pass"),
        },
        "by_mode": {},
    }

    for mode, rows in by_mode.items():
        summary["by_mode"][mode] = {
            "count": len(rows),
            "mode_accuracy": metric(rows, "mode_ok"),
            "grounding_ok_rate": metric(rows, "grounding_ok"),
            "drift_ok_rate": metric(rows, "drift_ok"),
            "signal_hit_rate": metric(rows, "signal_hit"),
            "overall_pass_rate": metric(rows, "overall_pass"),
        }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to answer_eval.v1.jsonl")
    parser.add_argument("--endpoint", required=True, help="Answer endpoint, e.g. http://127.0.0.1:8000/answer")
    parser.add_argument("--output-summary", required=True, help="Summary JSON output path")
    parser.add_argument("--output-details", required=True, help="Per-record JSONL output path")
    parser.add_argument("--allow-llm-answer", action="store_true", help="Call /answer with allow_llm_answer=true")
    args = parser.parse_args()

    rows = read_jsonl(Path(args.dataset))
    results = [evaluate_one(row, args.endpoint, args.allow_llm_answer) for row in rows]
    summary = summarize(results)

    Path(args.output_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_summary).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_jsonl(Path(args.output_details), results)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

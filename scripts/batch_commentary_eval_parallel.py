# -*- coding: utf-8 -*-
import json
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# =========================
# Config
DETAILS_DIR = Path("/root/persona-skill-factory/artifacts/evals")
DATASET_PATH = DETAILS_DIR / "answer_eval.v1.jsonl"
FAIL_SAMPLES_PATH = DETAILS_DIR / "commentary_drift_fail_samples.jsonl"

CORE_KEYWORDS = ["\u5206\u6790", "\u5224\u65ad", "\u8bc4\u8bba", "\u4f9d\u636e"]
MIN_TOKEN_THRESHOLD = 50
MAX_TOKENS_PER_SEGMENT = 2048
TEMPERATURE = 0.3

SEGMENTS = ["teaching", "bridge", "commentary", "persona"]
COMMENTARY_SUBSEGMENTS = ["\u5206\u6790", "\u5224\u65ad", "\u8bc4\u8bba", "\u4f9d\u636e"]
BATCH_SIZE = 20

ENDPOINT = "http://127.0.0.1:8000"  # 主服务 uvicorn 地址

# =========================
# Load dataset
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# =========================
# LLM 调用
def call_llm(query, mode):
    headers = {"Content-Type": "application/json"}
    payload = {
        "query": query,
        "mode": mode,
        "max_new_tokens": MAX_TOKENS_PER_SEGMENT,
        "temperature": TEMPERATURE,
        "allow_llm_answer": True
    }
    resp = requests.post(f"{ENDPOINT}/answer", json=payload, headers=headers)
    resp_json = resp.json()
    if "answer_text" in resp_json:
        return resp_json["answer_text"]
    else:
        print("DEBUG: LLM returned abnormal response:", resp_json)
        return ""

# =========================
# Eval 单线程处理单模式
def eval_mode(mode_name, samples):
    fail_samples = []
    lengths_stats = {seg: [] for seg in COMMENTARY_SUBSEGMENTS}

    for s in tqdm(samples, desc=f"Mode {mode_name}", ncols=100):
        eval_id = s.get("eval_id", "unknown")
        user_query = s.get("query", "")
        evidence_list = s.get("expected_evidence", [])

        segment_outputs = []

        if mode_name == "commentary":
            for j, seg_name in enumerate(COMMENTARY_SUBSEGMENTS):
                prompt = (
                    f"Generate {seg_name} segment.\n"
                    f"Must include keyword: {seg_name}\n"
                    f"Reference evidence: {evidence_list}\n"
                    f"Output complete sentences, minimum 50 tokens.\n"
                    f"Problem: {user_query}"
                )
                seg_output = call_llm(prompt, mode_name)
                segment_outputs.append(seg_output)
                lengths_stats[seg_name].append(len(seg_output.strip()))
        else:
            seg_output = call_llm(user_query, mode_name)
            segment_outputs.append(seg_output)
            lengths_stats["full"] = lengths_stats.get("full", []) + [len(seg_output.strip())]

        full_output = "\n".join(segment_outputs)
        keyword_hits = [kw for kw in CORE_KEYWORDS if kw in full_output]

        if len(full_output.strip()) < MIN_TOKEN_THRESHOLD or len(keyword_hits) < len(CORE_KEYWORDS):
            fail_samples.append({
                "eval_id": eval_id,
                "mode": mode_name,
                "output": full_output,
                "used_evidence_ids": evidence_list,
                "keyword_hits": keyword_hits,
                "length": len(full_output),
                "segment_lengths": {seg: lengths_stats[seg][-1] if lengths_stats[seg] else 0 for seg in COMMENTARY_SUBSEGMENTS}
            })

    for seg, lengths in lengths_stats.items():
        if lengths:
            avg_len = sum(lengths)/len(lengths)
            print(f"[{mode_name}] Segment {seg} avg length: {avg_len:.1f}, min: {min(lengths)}, max: {max(lengths)}")

    print(f"[{mode_name}] Completed. Fail samples: {len(fail_samples)}")
    return fail_samples

# =========================
# 并行运行
mode_samples_map = {m: [s for s in data if s.get("expected_mode","").lower() == m] for m in SEGMENTS}
all_fail_samples = []

with ThreadPoolExecutor(max_workers=4) as executor:
    future_to_mode = {executor.submit(eval_mode, mode, samples): mode for mode, samples in mode_samples_map.items()}
    for future in as_completed(future_to_mode):
        mode_name = future_to_mode[future]
        fails = future.result()
        all_fail_samples.extend(fails)

# =========================
# 写入 fail 样例
with open(FAIL_SAMPLES_PATH, "w", encoding="utf-8") as f:
    for s in all_fail_samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"\nAll commentary drift fail samples written to {FAIL_SAMPLES_PATH}")
print(f"Total fail samples: {len(all_fail_samples)}")

# =========================
# 汇总报表
summary = {mode: {seg: {"lengths": [], "fail_count": 0} for seg in COMMENTARY_SUBSEGMENTS} for mode in SEGMENTS}
for s in all_fail_samples:
    mode = s["mode"]
    segment_lengths = s.get("segment_lengths", {})
    for seg in COMMENTARY_SUBSEGMENTS:
        length = segment_lengths.get(seg, 0)
        summary[mode][seg]["lengths"].append(length)
        if length < MIN_TOKEN_THRESHOLD:
            summary[mode][seg]["fail_count"] += 1

print("\n=== Commentary Fail & Length Summary ===")
for mode in SEGMENTS:
    print(f"\nMode: {mode}")
    for seg in COMMENTARY_SUBSEGMENTS:
        lengths = summary[mode][seg]["lengths"]
        if lengths:
            avg_len = sum(lengths)/len(lengths)
            min_len = min(lengths)
            max_len = max(lengths)
            fail_count = summary[mode][seg]["fail_count"]
            fail_rate = fail_count / len(lengths)
            print(f"  Segment {seg}: avg={avg_len:.1f}, min={min_len}, max={max_len}, fail_count={fail_count}, fail_rate={fail_rate:.2%}")
        else:
            print(f"  Segment {seg}: No data")
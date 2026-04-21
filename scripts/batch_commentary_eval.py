# -*- coding: utf-8 -*-
import json
import requests
from pathlib import Path
import math

# =========================
# Config
# =========================
DETAILS_DIR = Path("/root/persona-skill-factory/artifacts/evals")
DATASET_PATH = DETAILS_DIR / "answer_eval.v1.jsonl"
ENDPOINT = "http://127.0.0.1:8001"  # vLLM 服务
BATCH_SIZE = 20
FAIL_SAMPLES_PATH = DETAILS_DIR / "commentary_drift_fail_samples.jsonl"

CORE_KEYWORDS = ["\u5206\u6790", "\u5224\u65ad", "\u8bc4\u8bba", "\u4f9d\u636e"]
MIN_TOKEN_THRESHOLD = 50
MAX_TOKENS_PER_SEGMENT = 2048
TEMPERATURE = 0.3
SEGMENTS = ["\u5206\u6790", "\u5224\u65ad", "\u8bc4\u8bba", "\u4f9d\u636e"]
MODEL_NAME = "Qwen/Qwen3-8B-AWQ"

# =========================
# Load commentary samples
# =========================
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    data = [
        json.loads(line) for line in f
        if json.loads(line).get("expected_mode", "").lower() == "commentary"
    ]

total = len(data)
batches = math.ceil(total / BATCH_SIZE)
print(f"Total commentary samples: {total}, Number of batches: {batches}")

all_fail_samples = []

# =========================
# Helper: call LLM
# =========================
def call_llm(prompt, mode="commentary"):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_new_tokens": MAX_TOKENS_PER_SEGMENT,
        "temperature": TEMPERATURE
    }
    resp = requests.post(f"{ENDPOINT}/v1/chat/completions", json=payload, headers=headers)
    resp_json = resp.json()
    # 兼容返回格式
    if "choices" in resp_json and resp_json["choices"]:
        return resp_json["choices"][0]["message"]["content"]
    elif "results" in resp_json and resp_json["results"]:
        return resp_json["results"][0].get("text", "")
    else:
        print("Unexpected LLM response:", resp_json)
        return ""

# =========================
# Run eval in batches
# =========================
for i in range(batches):
    batch_data = data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
    print(f"\nRunning batch {i+1}/{batches}...")

    for sample in batch_data:
        eval_id = sample.get("eval_id", "unknown")
        user_query = sample.get("query", "")
        evidence_list = sample.get("expected_evidence", [])

        segment_outputs = []
        segment_stats = []

        # 分段生成
        for j, seg_name in enumerate(SEGMENTS):
            evidence_for_seg = evidence_list[j::len(SEGMENTS)] if evidence_list else []
            prompt = (
                f"Generate the {seg_name} segment.\n"
                f"Must include keyword: {seg_name}\n"
                f"Reference the following evidence: {evidence_for_seg}\n"
                f"Output in complete sentences, minimum 50 tokens.\n"
                f"Do not output empty text.\n"
                f"Problem: {user_query}"
            )
            response_text = call_llm(prompt, mode="commentary")
            segment_outputs.append(response_text)

            # 单段统计
            seg_keywords = [kw for kw in CORE_KEYWORDS if kw in response_text]
            segment_stats.append({
                "length": len(response_text.strip()),
                "keyword_hits": seg_keywords
            })

        # 合并四段
        full_commentary = "\n".join(segment_outputs)

        # =========================
        # fail 判断：只看全文长度 + 全文关键词
        # =========================
        keyword_hits_full = [kw for kw in CORE_KEYWORDS if kw in full_commentary]
        fail_flag = len(full_commentary.strip()) < MIN_TOKEN_THRESHOLD or len(keyword_hits_full) < len(CORE_KEYWORDS)

        if fail_flag:
            all_fail_samples.append({
                "eval_id": eval_id,
                "prompt": user_query,
                "output": full_commentary,
                "used_evidence_ids": evidence_list,
                "keyword_hits": keyword_hits_full,
                "length": len(full_commentary),
                "segment_stats": segment_stats
            })

# =========================
# Write fail samples
# =========================
with open(FAIL_SAMPLES_PATH, "w", encoding="utf-8") as f_fail:
    for s in all_fail_samples:
        f_fail.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"\nAll commentary drift fail samples written to: {FAIL_SAMPLES_PATH}")
print(f"Total drift fail samples: {len(all_fail_samples)}")

# =========================
# Prompt template
# =========================
print("\n=== Suggested Prompt Template for Commentary ===")
print(
    "Please generate a complete commentary for the problem. Minimum 50 tokens.\n"
    "Must include the following keywords: \u5206\u6790, \u5224\u65ad, \u8bc4\u8bba, \u4f9d\u636e.\n"
    "Reference at least one evidence.\n"
    "Do not output empty text.\n"
    "Example output:\n"
    "[\u5206\u6790] ...\n"
    "[\u5224\u65ad] ...\n"
    "[\u8bc4\u8bba] ...\n"
    "[\u4f9d\u636e] ..."
)
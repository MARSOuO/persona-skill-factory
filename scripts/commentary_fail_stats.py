# -*- coding: utf-8 -*-
import json
from pathlib import Path
import statistics

# =========================
# 配置
# =========================
FAIL_SAMPLES_PATH = Path("/root/persona-skill-factory/artifacts/evals/commentary_drift_fail_samples.jsonl")

# =========================
# 读取 fail 样例
# =========================
fail_samples = []
with open(FAIL_SAMPLES_PATH, "r", encoding="utf-8") as f:
    for line in f:
        fail_samples.append(json.loads(line))

total_fail = len(fail_samples)
print(f"Total commentary fail samples: {total_fail}")

# =========================
# 分析每段长度与关键词
# =========================
segments = ["\u5206\u6790", "\u5224\u65ad", "\u8bc4\u8bba", "\u4f9d\u636e"]
segment_lengths = {seg: [] for seg in segments}
segment_fail_counts = {seg: 0 for seg in segments}

for s in fail_samples:
    for i, seg_name in enumerate(segments):
        # segment_fail_flags 列表与段序对应
        failed = s.get("segment_fail_flags", [True]*4)[i]
        output_lines = s.get("output", "").split("\n")
        if i < len(output_lines):
            length = len(output_lines[i].strip())
        else:
            length = 0
        segment_lengths[seg_name].append(length)
        if failed:
            segment_fail_counts[seg_name] += 1

# =========================
# 输出统计
# =========================
print("\n=== Commentary Segment Statistics ===")
for seg in segments:
    lengths = segment_lengths[seg]
    if lengths:
        avg_len = statistics.mean(lengths)
        min_len = min(lengths)
        max_len = max(lengths)
    else:
        avg_len = min_len = max_len = 0
    fail_count = segment_fail_counts[seg]
    print(f"{seg}: avg_len={avg_len:.1f}, min_len={min_len}, max_len={max_len}, fail_count={fail_count}")

# =========================
# Drift pass rate
# =========================
if total_fail > 0:
    drift_pass_rate = 1 - total_fail / total_fail  # 这里只是 fail 样例统计
    print(f"\nDrift pass rate (based on fail samples): {drift_pass_rate:.2%}")
else:
    print("\nAll samples passed drift check.")
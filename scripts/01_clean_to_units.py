from __future__ import annotations

import argparse
import re
from pathlib import Path

import jsonlines

from skill_factory.models import CleanSegment, ModeLabel, SourceSpan, UnitRecord


SENT_SPLIT_RE = re.compile(r"[。！？!?；;]\s*")
CLAUSE_SPLIT_RE = re.compile(r"[，、,]\s*")
MULTISPACE_RE = re.compile(r"\s+")

STYLE_KEYWORDS = {
    "persona": ["诡异", "怪谈", "离谱", "姐妹们", "笑死", "吓人", "不对劲", "直播", "弹幕"],
    "commentary": ["我不认同", "不太认同", "问题在于", "证据不够", "逻辑", "判断", "我认为", "我个人是会认为"],
    "bridge": ["其实不难", "先别急", "慢慢来", "先把话说明白", "没那么可怕", "举个例子", "打个比方"],
}

TEACHING_KEYWORDS = [
    "第一步", "第二步", "先", "再", "判断", "定义", "性质", "结论",
    "这道题", "很多同学", "做题", "步骤", "方法", "为什么",
    "讲", "解释", "例子", "举个例子", "总结", "导入", "运用"
]

COMMENTARY_KEYWORDS = [
    "我认为", "我不认同", "不太认同", "问题在于", "本质上", "其实不是",
    "危险", "太危险了", "逻辑", "证据不够", "我个人是会认为"
]

PERSONA_KEYWORDS = [
    "姐妹们", "笑死", "离谱", "怪谈", "弹幕", "直播", "上头", "整活"
]


def normalize_text(text: str) -> str:
    text = text.strip()
    text = MULTISPACE_RE.sub(" ", text)
    return text


def hard_split(text: str, max_len: int) -> list[str]:
    text = normalize_text(text)
    if len(text) <= max_len:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_len, len(text))
        chunks.append(text[start:end].strip())
        start = end
    return [c for c in chunks if c]


def merge_parts(parts: list[str], max_len: int) -> list[str]:
    chunks: list[str] = []
    buf = ""

    for part in parts:
        part = normalize_text(part)
        if not part:
            continue

        candidate = part if not buf else f"{buf}，{part}"
        if len(candidate) <= max_len:
            buf = candidate
        else:
            if buf:
                chunks.append(buf)
            if len(part) <= max_len:
                buf = part
            else:
                chunks.extend(hard_split(part, max_len))
                buf = ""

    if buf:
        chunks.append(buf)

    return chunks


def split_text(text: str, sent_max_len: int = 90, clause_max_len: int = 70) -> list[str]:
    text = normalize_text(text)
    if len(text) <= clause_max_len:
        return [text]

    sent_parts = [p.strip() for p in SENT_SPLIT_RE.split(text) if p.strip()]
    if not sent_parts:
        sent_parts = [text]

    sentence_chunks = merge_parts(sent_parts, sent_max_len)

    final_chunks: list[str] = []
    for chunk in sentence_chunks:
        if len(chunk) <= clause_max_len:
            final_chunks.append(chunk)
            continue

        clause_parts = [p.strip() for p in CLAUSE_SPLIT_RE.split(chunk) if p.strip()]
        if clause_parts:
            final_chunks.extend(merge_parts(clause_parts, clause_max_len))
        else:
            final_chunks.extend(hard_split(chunk, clause_max_len))

    cleaned = []
    for c in final_chunks:
        c = normalize_text(c)
        if c:
            cleaned.append(c)
    return cleaned if cleaned else [text]


def keyword_hits(text: str, keywords: list[str]) -> list[str]:
    return [kw for kw in keywords if kw in text]


def detect_style_signals(text: str, mode: ModeLabel) -> list[str]:
    hits = []
    hits.extend(keyword_hits(text, STYLE_KEYWORDS.get(mode.value, [])))
    return list(dict.fromkeys(hits))


def detect_teaching_signals(text: str) -> list[str]:
    return list(dict.fromkeys(keyword_hits(text, TEACHING_KEYWORDS)))


def detect_commentary_signals(text: str) -> list[str]:
    return list(dict.fromkeys(keyword_hits(text, COMMENTARY_KEYWORDS)))


def detect_persona_signals(text: str) -> list[str]:
    return list(dict.fromkeys(keyword_hits(text, PERSONA_KEYWORDS)))


def infer_router_mode(corpus_mode: ModeLabel, text: str) -> ModeLabel:
    teaching_hits = len(detect_teaching_signals(text))
    commentary_hits = len(detect_commentary_signals(text))
    persona_hits = len(detect_persona_signals(text))

    if corpus_mode == ModeLabel.teaching:
        if commentary_hits >= 2:
            return ModeLabel.commentary
        return ModeLabel.teaching

    if corpus_mode == ModeLabel.commentary:
        return ModeLabel.commentary

    if corpus_mode == ModeLabel.persona:
        if teaching_hits >= 3:
            return ModeLabel.bridge
        return ModeLabel.persona

    if corpus_mode == ModeLabel.bridge:
        if teaching_hits >= 2 and teaching_hits >= commentary_hits:
            return ModeLabel.teaching
        if commentary_hits >= 2 and commentary_hits > teaching_hits:
            return ModeLabel.commentary
        if persona_hits >= 2:
            return ModeLabel.persona
        return ModeLabel.bridge

    if teaching_hits >= 2:
        return ModeLabel.teaching
    if commentary_hits >= 2:
        return ModeLabel.commentary
    if persona_hits >= 2:
        return ModeLabel.persona
    return corpus_mode


def is_extractable(text: str) -> bool:
    return len(text) >= 10


def build_unit_id(source_file: str, segment_id: str, idx: int) -> str:
    stem = Path(source_file).stem
    return f"unit_{stem}_{segment_id}_{idx:02d}"


def convert_segment(seg: CleanSegment) -> list[UnitRecord]:
    chunks = split_text(seg.text)
    units: list[UnitRecord] = []

    cursor = 0
    for idx, chunk in enumerate(chunks, start=1):
        start_char = seg.text.find(chunk, cursor)
        if start_char < 0:
            start_char = cursor
        end_char = start_char + len(chunk)
        cursor = end_char

        normalized = normalize_text(chunk)
        router_mode = infer_router_mode(seg.corpus_mode, normalized)
        style_signals = detect_style_signals(normalized, seg.corpus_mode)
        teaching_signals = detect_teaching_signals(normalized)
        commentary_signals = detect_commentary_signals(normalized)
        persona_signals = detect_persona_signals(normalized)

        unit = UnitRecord(
            unit_id=build_unit_id(seg.source_file, seg.segment_id, idx),
            corpus_mode=seg.corpus_mode,
            router_mode=router_mode,
            source=SourceSpan(
                source_file=seg.source_file,
                segment_id=seg.segment_id,
                start_char=start_char,
                end_char=end_char,
                start_sec=seg.start_sec,
                end_sec=seg.end_sec,
            ),
            text=chunk,
            normalized_text=normalized,
            style_signals=style_signals,
            teaching_signals=teaching_signals,
            tags=[
                seg.corpus_mode.value,
                router_mode.value,
                *commentary_signals,
                *persona_signals,
            ],
            extractable=is_extractable(normalized),
            confidence=0.8,
            metadata={
                "speaker": seg.speaker,
                "title": seg.metadata.get("title"),
                "paragraph_id": seg.metadata.get("paragraph_id"),
                "filename": seg.metadata.get("filename"),
            },
        )
        units.append(unit)

    return units


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to clean JSONL")
    parser.add_argument("--output", required=True, help="Path to units JSONL")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_segments = 0
    total_units = 0

    with jsonlines.open(input_path, "r") as reader, jsonlines.open(output_path, "w") as writer:
        for row in reader:
            seg = CleanSegment.model_validate(row)
            total_segments += 1
            units = convert_segment(seg)
            for unit in units:
                writer.write(unit.model_dump(mode="json"))
                total_units += 1

    print(f"done: segments={total_segments}, units={total_units}, output={output_path}")


if __name__ == "__main__":
    main()

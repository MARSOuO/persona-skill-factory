from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import jsonlines

from skill_factory.models import ModeLabel, RouterLabelRecord, UnitRecord


TEACHING_STEP_KWS = ["第一步", "第二步", "先", "再", "步骤"]
TEACHING_METHOD_KWS = ["定义", "性质", "结论", "判断", "方法", "做题", "这道题"]
TEACHING_EXAMPLE_KWS = ["举个例子", "打个比方", "例子", "类比"]
TEACHING_WRAPUP_KWS = ["总结", "归纳", "运用", "导入", "解释"]

COMMENTARY_JUDGMENT_KWS = ["我认为", "我个人是会认为", "我不认同", "不太认同"]
COMMENTARY_PROBLEM_KWS = ["问题在于", "本质上", "其实不是", "逻辑", "证据不够", "危险", "太危险了"]

PERSONA_INTERACTION_KWS = ["姐妹们", "笑死", "离谱", "直播", "弹幕", "上头", "整活"]
BRIDGE_LIGHT_KWS = ["其实不难", "先别急", "慢慢来", "举个例子", "打个比方", "先把话说明白"]


def has_any(text: str, keywords: list[str]) -> bool:
    return any(kw in text for kw in keywords)


def unique_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def build_sample_id(unit_id: str, query: str) -> str:
    h = hashlib.md5(f"{unit_id}|{query}".encode("utf-8")).hexdigest()[:10]
    return f"router_{h}"


def candidate_modes_for(gold: ModeLabel) -> list[ModeLabel]:
    # V1 里保留全部四类，顺序把 gold 放前面，便于后续训练/评估
    all_modes = [ModeLabel.teaching, ModeLabel.bridge, ModeLabel.commentary, ModeLabel.persona]
    ordered = [gold] + [m for m in all_modes if m != gold]
    return ordered


def infer_route_reason(unit: UnitRecord) -> str:
    reasons = [f"router_mode={unit.router_mode.value}", f"corpus_mode={unit.corpus_mode.value}"]

    if unit.teaching_signals:
        reasons.append(f"teaching_signals={','.join(unit.teaching_signals[:4])}")

    if unit.style_signals:
        reasons.append(f"style_signals={','.join(unit.style_signals[:4])}")

    if unit.router_mode == ModeLabel.commentary:
        reasons.append("contains stance/judgment pattern")
    elif unit.router_mode == ModeLabel.teaching:
        reasons.append("contains explanation/method progression")
    elif unit.router_mode == ModeLabel.bridge:
        reasons.append("contains light-bridge or soft-intro pattern")
    elif unit.router_mode == ModeLabel.persona:
        reasons.append("contains interaction/persona expression")

    return "; ".join(reasons)


def build_query(unit: UnitRecord) -> str:
    text = unit.normalized_text
    mode = unit.router_mode

    if mode == ModeLabel.teaching:
        if has_any(text, TEACHING_STEP_KWS):
            return "这部分内容能不能分步骤给我讲？"
        if has_any(text, TEACHING_METHOD_KWS):
            return "这种题是不是应该先判断，再选方法？"
        if has_any(text, TEACHING_EXAMPLE_KWS):
            return "能不能先举个例子，再解释这个知识点？"
        if has_any(text, TEACHING_WRAPUP_KWS):
            return "讲完之后能不能顺手帮我总结一下？"
        return "这个知识点你能带着我慢慢讲明白吗？"

    if mode == ModeLabel.commentary:
        if has_any(text, COMMENTARY_JUDGMENT_KWS):
            return "你对这件事到底怎么看，能直接表态吗？"
        if has_any(text, COMMENTARY_PROBLEM_KWS):
            return "这套说法的问题本质到底在哪？"
        return "你能不能别只复述，直接说你的判断？"

    if mode == ModeLabel.bridge:
        if has_any(text, TEACHING_EXAMPLE_KWS + BRIDGE_LIGHT_KWS):
            return "能不能别太硬核，轻松一点带我理解这个内容？"
        return "这个东西你能不能先温和地带我入门？"

    # persona
    if has_any(text, PERSONA_INTERACTION_KWS):
        return "别太正经，按你平时那种互动口气陪我聊聊。"
    return "用你那种有人味、带互动感的方式跟我说。"


def quality_ok(unit: UnitRecord) -> bool:
    if not unit.extractable:
        return False
    text_len = len(unit.normalized_text)
    if text_len < 10:
        return False
    if text_len > 120:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="units jsonl")
    parser.add_argument("--output", required=True, help="router_labels jsonl")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_units = 0
    total_labels = 0

    with jsonlines.open(input_path, "r") as reader, jsonlines.open(output_path, "w") as writer:
        for row in reader:
            unit = UnitRecord.model_validate(row)
            total_units += 1

            if not quality_ok(unit):
                continue

            query = build_query(unit)
            record = RouterLabelRecord(
                sample_id=build_sample_id(unit.unit_id, query),
                query=query,
                gold_mode=unit.router_mode,
                candidate_modes=candidate_modes_for(unit.router_mode),
                route_reason=infer_route_reason(unit),
                evidence_unit_ids=[unit.unit_id],
                confidence=0.82 if unit.router_mode != ModeLabel.bridge else 0.76,
                metadata={
                    "source_file": unit.source.source_file,
                    "segment_id": unit.source.segment_id,
                    "paragraph_id": unit.metadata.get("paragraph_id"),
                    "filename": unit.metadata.get("filename"),
                    "corpus_mode": unit.corpus_mode.value,
                    "router_mode": unit.router_mode.value,
                    "text_preview": unit.text[:120],
                    "style_signals": unit.style_signals,
                    "teaching_signals": unit.teaching_signals,
                },
            )
            writer.write(record.model_dump(mode="json"))
            total_labels += 1

    print(f"done: units={total_units}, router_labels={total_labels}, output={output_path}")


if __name__ == "__main__":
    main()

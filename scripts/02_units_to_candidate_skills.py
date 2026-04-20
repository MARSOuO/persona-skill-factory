from __future__ import annotations

import argparse
from pathlib import Path

import jsonlines

from skill_factory.models import CandidateSkillRecord, ModeLabel, UnitRecord


TEACHING_STEP_KWS = ["第一步", "第二步", "先", "再", "步骤"]
TEACHING_METHOD_KWS = ["定义", "性质", "结论", "判断", "方法", "做题", "这道题"]
TEACHING_EXAMPLE_KWS = ["举个例子", "打个比方", "例子", "类比"]
TEACHING_WRAPUP_KWS = ["总结", "归纳", "最后", "运用", "导入", "解释"]

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


def should_keep(unit: UnitRecord) -> bool:
    if not unit.extractable:
        return False

    if unit.router_mode in {ModeLabel.teaching, ModeLabel.commentary}:
        return True

    if unit.router_mode == ModeLabel.bridge:
        return bool(unit.teaching_signals or unit.style_signals or len(unit.text) >= 18)

    if unit.router_mode == ModeLabel.persona:
        return bool(unit.style_signals or has_any(unit.text, PERSONA_INTERACTION_KWS)) and len(unit.text) >= 18

    return False


def infer_skill(unit: UnitRecord) -> tuple[str, str, list[str], list[str], list[str], list[str], list[str]]:
    text = unit.normalized_text
    mode = unit.router_mode

    if mode == ModeLabel.teaching:
        if has_any(text, TEACHING_STEP_KWS):
            return (
                "分步骤推进讲解",
                "把问题拆成若干步骤，按顺序推进，而不是一次性抛结论。",
                ["用户面对新题、复杂题、陌生知识点时，不知道从哪里下手"],
                ["信息尚未分类，容易一上来就硬算或乱套方法"],
                ["先给出起手判断点", "再按步骤推进", "最后补一个收束或提醒"],
                ["不要一上来直接给最终答案", "不要跳步骤"],
                ["没有步骤意识的闲聊段落", "只有情绪表达、没有方法动作的文本"],
            )
        if has_any(text, TEACHING_METHOD_KWS):
            return (
                "先判断再选方法",
                "先判断题目条件和问题类型，再选择定义、性质、结论或具体方法。",
                ["用户看到题就想直接代公式或直接求值"],
                ["缺少题型判断，方法选择过早"],
                ["先识别已知条件", "判断该用哪类工具", "再进入计算或推导"],
                ["不要不经判断就套模板", "不要把结论当前提"],
                ["只有知识点描述、没有方法选择动作的文本"],
            )
        if has_any(text, TEACHING_EXAMPLE_KWS):
            return (
                "用类比或例子导入知识点",
                "先用类比、比喻或例子降低理解门槛，再进入正式解释。",
                ["用户对抽象概念没有抓手，直接解释容易听不进去"],
                ["知识点过抽象，缺少感性入口"],
                ["先给熟悉场景或比喻", "再说概念本体", "最后回到正式用法"],
                ["不要一开始就堆术语", "不要类比完却不回到正题"],
                ["纯定义式讲解，没有桥接动作的文本"],
            )
        if has_any(text, TEACHING_WRAPUP_KWS):
            return (
                "讲解后做归纳收束",
                "讲完后补一个总结、运用提醒或收束，帮助形成完整闭环。",
                ["一段解释结束后，用户仍不知道该如何落到做题或应用"],
                ["只讲过程，没有收尾，难以形成稳定记忆"],
                ["提炼一句总规则", "指出使用场景", "提醒常见误区"],
                ["不要讲完就停", "不要只重复原文不做提炼"],
                ["只有开场导入，没有总结闭环的文本"],
            )
        return (
            "教学解释推进",
            "围绕知识点进行解释、拆解和推进，让用户跟得上理解节奏。",
            ["用户对内容陌生，需要被带着走"],
            ["需要先建立理解框架，再进入细节"],
            ["解释概念", "拆开信息", "补过渡语", "控制节奏"],
            ["不要只抛结论", "不要解释过密导致跟丢"],
            ["没有明确教学意图的碎片表达"],
        )

    if mode == ModeLabel.commentary:
        if has_any(text, COMMENTARY_JUDGMENT_KWS):
            return (
                "明确表态式判断输出",
                "先亮明自己的判断立场，再展开理由，而不是假装中立。",
                ["面对观点、现象、说法，需要快速给出态度"],
                ["用户想知道你的判断，不只是材料复述"],
                ["先给判断", "再给理由", "最后补边界或提醒"],
                ["不要绕半天不表态", "不要只有态度没有依据"],
                ["只是复述材料、没有立场的文本"],
            )
        if has_any(text, COMMENTARY_PROBLEM_KWS):
            return (
                "指出问题本质",
                "抓住说法里的关键漏洞、本质矛盾或证据不足之处。",
                ["用户给出一个看似有道理的说法，希望判断是否成立"],
                ["表面说法顺耳，但底层逻辑有问题"],
                ["指出问题点", "解释为什么有问题", "给出更稳的看法"],
                ["不要只说不对却不解释", "不要攻击人而不攻击论证"],
                ["单纯情绪宣泄，没有问题定位的文本"],
            )
        return (
            "评论性判断输出",
            "针对现象、说法或观点做解释型判断，不止描述，还要定性。",
            ["需要对一个对象做判断性输出"],
            ["用户要的不是复读，而是带立场的分析"],
            ["定性", "解释", "补依据"],
            ["不要只有复述没有判断"],
            ["纯陪伴式闲聊文本"],
        )

    if mode == ModeLabel.bridge:
        if has_any(text, TEACHING_EXAMPLE_KWS + BRIDGE_LIGHT_KWS):
            return (
                "轻松包装式知识导入",
                "用轻松口气、比喻或玩梗先把用户带进来，再过渡到知识点。",
                ["知识点本身不难，但直接讲会干、会硬"],
                ["用户进入门槛高，需要先降低心理阻力"],
                ["先用轻松表达破冰", "再切知识点", "最后落回方法或结论"],
                ["不要只顾玩梗忘了回到知识点", "不要让包装压过内容"],
                ["只有教学、没有桥接动作的文本", "只有玩梗、没有知识推进的文本"],
            )
        return (
            "轻松过渡式讲解",
            "在轻松口气和知识表达之间搭桥，降低理解阻力。",
            ["用户需要被温和带入，而不是被硬塞信息"],
            ["直接进入正题可能会让人退缩或走神"],
            ["先安抚", "再引入", "再逐步推进"],
            ["不要全程悬浮在气氛层", "不要忘记落到内容"],
            ["纯人设聊天文本", "纯硬核讲题文本"],
        )

    # persona
    return (
        "人设化互动表达",
        "用固定口气、互动感和人设风格维持在场感与陪伴感。",
        ["需要拉近距离、制造陪伴氛围、维持直播感"],
        ["内容本身不一定以知识推进为主，但需要强烈在场感"],
        ["保持口气一致", "用互动词维持现场感", "适度放入人设化表达"],
        ["不要突然切成生硬说明文", "不要过度空转无内容"],
        ["纯知识解释文本", "纯判断分析文本"],
    )


def build_candidate(unit: UnitRecord) -> CandidateSkillRecord:
    name, description, trigger, diagnosis, response_plan, avoid, negative_cases = infer_skill(unit)

    return CandidateSkillRecord(
        skill_id=f"cand_{unit.unit_id}",
        name=name,
        mode=unit.router_mode,
        description=description,
        trigger=trigger,
        diagnosis=diagnosis,
        response_plan=response_plan,
        avoid=avoid,
        negative_cases=negative_cases,
        evidence_units=[unit.unit_id],
        evidence_quotes=[unit.text[:160]],
        confidence=0.72 if unit.router_mode == ModeLabel.bridge else 0.76,
        metadata={
            "source_file": unit.source.source_file,
            "segment_id": unit.source.segment_id,
            "paragraph_id": unit.metadata.get("paragraph_id"),
            "filename": unit.metadata.get("filename"),
            "corpus_mode": unit.corpus_mode.value,
            "router_mode": unit.router_mode.value,
            "style_signals": unit.style_signals,
            "teaching_signals": unit.teaching_signals,
            "tags": unique_keep_order(unit.tags),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to units JSONL")
    parser.add_argument("--output", required=True, help="Path to candidate skills JSONL")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_units = 0
    kept_units = 0

    with jsonlines.open(input_path, "r") as reader, jsonlines.open(output_path, "w") as writer:
        for row in reader:
            unit = UnitRecord.model_validate(row)
            total_units += 1

            if not should_keep(unit):
                continue

            cand = build_candidate(unit)
            writer.write(cand.model_dump(mode="json"))
            kept_units += 1

    print(f"done: units={total_units}, candidate_skills={kept_units}, output={output_path}")


if __name__ == "__main__":
    main()

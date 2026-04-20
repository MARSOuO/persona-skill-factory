from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

from app.domain.models import CanonicalSkill
from app.schemas.route_and_plan import ModeScore, PredictedMode

TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]+")


def _has_any(text: str, phrases: list[str]) -> list[str]:
    return [p for p in phrases if p and p in text]


MODE_KEYWORDS: dict[str, list[str]] = {
    "teaching": [
        "怎么", "如何", "为什么", "原理", "步骤", "分步骤", "讲清楚", "拆开", "推导", "方法",
        "分析", "框架", "系统讲", "讲解", "解释", "题", "知识点",
    ],
    "bridge": [
        "温和", "轻松", "好懂", "带我入门", "先带一下", "慢慢来", "别太硬", "别上来太硬",
        "先聊聊", "先顺一下", "过渡", "包装", "引我进去", "直观", "先给我一个感觉",
    ],
    "commentary": [
        "本质", "问题", "怎么看", "判断", "锐评", "表态", "离谱", "到底", "核心矛盾",
        "值不值得", "是不是", "毛病", "误区", "关键问题",
    ],
    "persona": [
        "有人味", "互动感", "陪我", "跟我说", "直播口气", "怪谈", "聊天", "像平时那样",
        "松弛一点", "接地气", "有梗", "唠", "宝宝", "姐妹", "晚安",
    ],
}

FEATURE_GROUPS: dict[str, list[str]] = {
    "teaching_structure": [
        "分步骤", "一步一步", "步骤", "拆开", "讲清楚", "系统讲", "推导", "顺序", "框架", "方法",
    ],
    "teaching_explain": [
        "为什么", "原理", "怎么", "如何", "分析", "解释", "理解", "知识点", "题",
    ],
    "teaching_method_flow": [
        "先判断", "再选方法", "选方法", "这种题", "解题", "题型", "下手", "做题",
        "判断题型", "判断再做", "先看题型", "先定方法",
    ],
    "bridge_softness": [
        "温和", "轻松", "柔和", "慢慢来", "别太硬", "别上来太硬", "好懂", "轻一点",
    ],
    "bridge_intro": [
        "带我入门", "先带一下", "先聊聊", "引我进去", "先顺一下", "过渡", "包装", "先给我一个感觉",
    ],
    "commentary_judgment": [
        "怎么看", "判断", "表态", "值不值得", "是不是", "到底", "我觉得",
    ],
    "commentary_critique": [
        "本质", "问题", "核心矛盾", "离谱", "毛病", "误区", "关键问题", "锐评",
    ],
    "commentary_stance": [
        "怎么看", "表态", "值不值得", "离谱", "到底", "本质", "核心矛盾", "毛病", "误区",
    ],
    "persona_interaction": [
        "有人味", "互动感", "陪我", "跟我说", "像平时那样", "聊天", "唠", "陪伴",
    ],
    "persona_style": [
        "直播口气", "怪谈", "松弛一点", "接地气", "有梗", "宝宝", "姐妹", "晚安",
    ],
}


class RouterService:
    version = "router.mode-boundary.v4"

    def __init__(self, skills: Iterable[CanonicalSkill]):
        self.skills = list(skills)
        self.skills_by_mode: dict[str, list[CanonicalSkill]] = defaultdict(list)
        for skill in self.skills:
            self.skills_by_mode[skill.mode].append(skill)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())

    def _best_skill_trigger_match(self, query: str) -> dict[str, tuple[int, str]]:
        best: dict[str, tuple[int, str]] = {
            "teaching": (0, ""),
            "bridge": (0, ""),
            "commentary": (0, ""),
            "persona": (0, ""),
        }
        for mode, skills in self.skills_by_mode.items():
            best_hits = 0
            best_name = ""
            for skill in skills:
                hits = sum(1 for t in skill.triggers if t and t in query)
                if hits > best_hits:
                    best_hits = hits
                    best_name = skill.name
            best[mode] = (best_hits, best_name)
        return best

    def predict(self, query: str) -> PredictedMode:
        q = self._normalize(query)
        tokens = self.tokenize(q)
        token_set = set(tokens)

        scores: dict[str, float] = {
            "teaching": 0.0,
            "bridge": 0.0,
            "commentary": 0.0,
            "persona": 0.0,
        }
        reasons: dict[str, list[str]] = {mode: [] for mode in scores}

        for mode, keywords in MODE_KEYWORDS.items():
            hits = [kw for kw in keywords if kw in q or kw.lower() in token_set]
            if hits:
                scores[mode] += 0.8 * len(hits)
                reasons[mode].append(f"命中模式关键词: {', '.join(hits[:4])}")

        feature_hits = {name: _has_any(q, phrases) for name, phrases in FEATURE_GROUPS.items()}

        ts = len(feature_hits["teaching_structure"])
        te = len(feature_hits["teaching_explain"])
        tm = len(feature_hits["teaching_method_flow"])

        bs = len(feature_hits["bridge_softness"])
        bi = len(feature_hits["bridge_intro"])

        cj = len(feature_hits["commentary_judgment"])
        cc = len(feature_hits["commentary_critique"])
        cs = len(feature_hits["commentary_stance"])

        pi = len(feature_hits["persona_interaction"])
        ps = len(feature_hits["persona_style"])

        if ts:
            scores["teaching"] += 1.4 * ts
            reasons["teaching"].append(
                f"结构化解释意图强: {', '.join(feature_hits['teaching_structure'][:4])}"
            )
        if te:
            scores["teaching"] += 0.9 * te
            reasons["teaching"].append(
                f"解释/求方法意图: {', '.join(feature_hits['teaching_explain'][:4])}"
            )
        if tm:
            scores["teaching"] += 1.8 * tm
            reasons["teaching"].append(
                f"方法流程信号: {', '.join(feature_hits['teaching_method_flow'][:4])}"
            )

        if bs:
            scores["bridge"] += 1.5 * bs
            reasons["bridge"].append(
                f"柔和导入信号: {', '.join(feature_hits['bridge_softness'][:4])}"
            )
        if bi:
            scores["bridge"] += 1.3 * bi
            reasons["bridge"].append(
                f"过渡/带入信号: {', '.join(feature_hits['bridge_intro'][:4])}"
            )

        if cj:
            scores["commentary"] += 1.5 * cj
            reasons["commentary"].append(
                f"判断/表态意图: {', '.join(feature_hits['commentary_judgment'][:4])}"
            )
        if cc:
            scores["commentary"] += 1.6 * cc
            reasons["commentary"].append(
                f"批评/指出问题意图: {', '.join(feature_hits['commentary_critique'][:4])}"
            )
        if cs:
            scores["commentary"] += 0.8 * cs
            reasons["commentary"].append(
                f"立场/锐评信号: {', '.join(feature_hits['commentary_stance'][:4])}"
            )

        if pi:
            scores["persona"] += 1.5 * pi
            reasons["persona"].append(
                f"互动/陪伴意图: {', '.join(feature_hits['persona_interaction'][:4])}"
            )
        if ps:
            scores["persona"] += 1.4 * ps
            reasons["persona"].append(
                f"人设/口气风格意图: {', '.join(feature_hits['persona_style'][:4])}"
            )

        best_skill_matches = self._best_skill_trigger_match(q)
        for mode, (hit_count, skill_name) in best_skill_matches.items():
            if hit_count > 0:
                bonus = min(1.2, 0.45 * hit_count)
                scores[mode] += bonus
                reasons[mode].append(f"命中该 mode 最相关 skill: {skill_name}")

        if cj + cc + cs >= 2 and tm == 0:
            scores["commentary"] += 2.2
            reasons["commentary"].append("覆盖规则: 明显是求判断/求锐评，commentary 优先")

        if bs + bi >= 2 and ts <= 1 and pi == 0 and cj == 0:
            scores["bridge"] += 2.0
            scores["teaching"] -= 0.8
            reasons["bridge"].append("覆盖规则: 柔和导入意图成立，bridge 优先于 teaching")

        if pi + ps >= 2 and ts <= 1 and cj == 0:
            scores["persona"] += 2.2
            scores["teaching"] -= 1.0
            reasons["persona"].append("覆盖规则: 互动/人设意图成立，persona 优先于 teaching")

        if tm >= 1 and ("方法" in q or "这种题" in q or "步骤" in q or "解题" in q):
            scores["teaching"] += 2.0
            scores["commentary"] -= 1.2
            reasons["teaching"].append("覆盖规则: 判断出现在解题流程中，teaching 优先")
            reasons["commentary"].append("去偏置: ‘判断/是不是’ 属于方法流程上下文，下调 commentary")

        if ts >= 2:
            scores["teaching"] += 1.6
            reasons["teaching"].append("覆盖规则: 强结构化讲解成立，teaching 收束")

        if ts == 0:
            if bs + bi > 0:
                scores["teaching"] -= 0.6
                reasons["teaching"].append("去偏置: 存在 bridge 信号，teaching 下调")
            if pi + ps > 0:
                scores["teaching"] -= 0.8
                reasons["teaching"].append("去偏置: 存在 persona 信号，teaching 下调")
            if cj + cc > 0 and tm == 0:
                scores["teaching"] -= 0.8
                reasons["teaching"].append("去偏置: 存在 commentary 信号，teaching 下调")

        for mode in scores:
            scores[mode] = round(max(scores[mode], 0.0), 4)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_mode, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = max(best_score - second_score, 0.0)

        confidence = round(min(0.99, 0.45 + 0.06 * best_score + 0.10 * margin), 4)

        candidates = [
            ModeScore(mode=mode, score=round(score, 4), reasons=reasons[mode][:3])
            for mode, score in ranked
        ]

        return PredictedMode(
            label=best_mode,
            confidence=confidence,
            candidates=candidates,
            router_version=self.version,
        )

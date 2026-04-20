from __future__ import annotations

import re

STOPWORDS = {
    "\u8fd9\u4e2a",  # 这个
    "\u90a3\u4e2a",  # 那个
    "\u4e00\u4e0b",  # 一下
    "\u4e00\u4e0b\u5b50",  # 一下子
    "\u53ef\u4ee5",  # 可以
    "\u80fd\u4e0d\u80fd",  # 能不能
    "\u4e0d\u80fd\u4e0d\u80fd",  # 不能不能
    "\u6211",  # 我
    "\u4f60",  # 你
    "\u554a",  # 啊
    "\u5462",  # 呢
    "\u5417",  # 吗
    "\u5440",  # 呀
    "\u5427",  # 吧
    "\u5c31",  # 就
    "\u8fd8",  # 还
    "\u5148",  # 先
    "\u518d",  # 再
    "\u7ed9\u6211",  # 给我
    "\u4e00\u4e0b\u5427",  # 一下吧
    "\u65b9\u5f0f",  # 方式
    "\u4e1c\u897f",  # 东西
    "\u5185\u5bb9",  # 内容
    "\u76f4\u63a5",  # 直接
    "\u5c31\u662f",  # 就是
    "\u7684\u8bdd",  # 的话
}

MIXED_RE = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+")


def tokenize_zh_en(text: str, min_n: int = 2, max_n: int = 4) -> list[str]:
    pieces = [m.group(0).lower() for m in MIXED_RE.finditer(text)]
    out: list[str] = []
    seen: set[str] = set()

    for piece in pieces:
        # 英文/数字
        if re.fullmatch(r"[a-zA-Z0-9_]+", piece):
            if piece not in STOPWORDS and piece not in seen:
                out.append(piece)
                seen.add(piece)
            continue

        # 中文：切成 2~4 字 n-gram
        n = len(piece)
        for k in range(min_n, min(max_n, n) + 1):
            for i in range(0, n - k + 1):
                gram = piece[i:i + k]
                if gram in STOPWORDS:
                    continue
                if gram not in seen:
                    out.append(gram)
                    seen.add(gram)

    return out

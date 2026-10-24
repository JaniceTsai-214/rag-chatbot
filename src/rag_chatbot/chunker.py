"""切段：以字元數計，優先在段落、句號處斷開，不跨 Section（設計見 docs/02-design.md §4.1）。"""

from __future__ import annotations

from collections.abc import Sequence

from rag_chatbot.models import Chunk, Section

# 斷開位置的優先順序：空行 > 換行 > 句末 > 逗號/空白
_BREAKS: tuple[tuple[str, ...], ...] = (
    ("\n\n",),
    ("\n",),
    ("。", "！", "？", "；", "!", "?", ";", ". "),
    ("，", "、", ",", " "),
)


def chunk_sections(
    sections: Sequence[Section],
    chunk_size: int,
    chunk_overlap: int,
    split: bool = True,
) -> list[Chunk]:
    """將 Section 切成 Chunk，chunk_index 從 0 連續編號。

    split=False 時每個 Section 原樣成為一段（FAQ 使用）。
    """
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_size 必須大於 0，chunk_overlap 必須介於 0 與 chunk_size 之間")

    chunks: list[Chunk] = []
    for section in sections:
        pieces = _split(section.text, chunk_size, chunk_overlap) if split else [section.text.strip()]
        for text in pieces:
            if text:
                chunks.append(Chunk(len(chunks), text, section.page, section.heading))
    return chunks


def _split(text: str, size: int, overlap: int) -> list[str]:
    text = text.strip()
    if len(text) <= size:
        return [text]

    pieces = []
    start = 0
    while True:
        end = start + size
        if end >= len(text):
            pieces.append(text[start:].strip())
            break
        cut = _find_cut(text, start + size // 2, end)
        pieces.append(text[start:cut].strip())
        start = _overlap_start(text, max(cut - overlap, start + 1), cut)
    return pieces


def _overlap_start(text: str, low: int, cut: int) -> int:
    """重疊部分盡量從完整句子開始：在 text[low:cut] 中找第一個句末斷點，從其後開始。"""
    found = [pos + len(m) for group in _BREAKS[:3] for m in group if (pos := text.find(m, low, cut)) >= 0]
    candidates = [p for p in found if p < cut]
    return min(candidates) if candidates else low


def _find_cut(text: str, low: int, high: int) -> int:
    """在 text[low:high] 中找最後一個優先度最高的斷點，回傳斷點之後的位置；找不到則硬切在 high。"""
    for markers in _BREAKS:
        found = [(pos, m) for m in markers if (pos := text.rfind(m, low, high)) >= 0]
        if found:
            pos, marker = max(found)
            return pos + len(marker)
    return high

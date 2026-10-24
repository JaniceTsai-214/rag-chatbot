"""核心資料類別（設計見 docs/02-design.md §7）。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Section:
    """Loader 輸出的文字區塊，帶有位置資訊。"""

    text: str
    page: int | None = None
    heading: str | None = None


@dataclass(frozen=True)
class Chunk:
    """切段後的段落，尚未向量化。"""

    chunk_index: int
    text: str
    page: int | None = None
    heading: str | None = None


@dataclass(frozen=True)
class Source:
    """回答依據：檢索到的段落與相似度。"""

    chunk_id: int
    collection: str
    filename: str
    page: int | None
    heading: str | None
    text: str
    score: float


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[Source]
    refused: bool  # 是否因檢索不到而拒答
    model: str
    prompt_version: int
    latency_ms: int

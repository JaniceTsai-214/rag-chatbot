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


ANSWERED = "answered"  # 回答
REFUSED = "refused"  # 拒答（門檻擋下，或模型判斷文件中沒有答案）
GREETING = "greeting"  # 問候、閒聊、套話的固定回覆


@dataclass(frozen=True)
class Answer:
    kind: str
    text: str
    sources: list[Source]  # 送給模型的段落；門檻拒答與問候時為空
    model: str | None  # 未呼叫 LLM 時為 None
    prompt_version: int | None
    latency_ms: int
    conversation_id: int
    top_score: float | None = None  # 語意分數最高值；問候時為 None

    @property
    def refused(self) -> bool:
        return self.kind == REFUSED

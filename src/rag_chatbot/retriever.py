"""混合檢索 + 門檻 + 參考資料長度上限（設計見 docs/02-design.md §4.2 步驟 4～6）。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from rag_chatbot.embedder import Embedder
from rag_chatbot.keyword_index import KeywordIndex
from rag_chatbot.models import Source
from rag_chatbot.vector_store import VectorStore, cosine_scores

RRF_K = 60  # Reciprocal Rank Fusion 的常數（常用值）
TRUNCATED_MARK = "…"


@dataclass(frozen=True)
class RetrievalResult:
    sources: list[Source]  # 依混合排名；score 為語意分數；未通過門檻時為空
    top_score: float | None  # 語意分數最高值；沒有任何段落時為 None
    passed: bool  # top_score 是否達到門檻


@dataclass
class _Corpus:
    chunks: list[Source]
    matrix: np.ndarray
    keywords: KeywordIndex


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder
        self._cache: dict[tuple[str, ...], tuple[int, _Corpus]] = {}

    def retrieve(
        self,
        query: str,
        collections: Sequence[str],
        top_k: int,
        min_score: float,
        max_context_chars: int | None = None,
    ) -> RetrievalResult:
        """max_context_chars 為 None 時不限長度（search 指令用）。"""
        if top_k <= 0:
            raise ValueError("top_k 必須大於 0")
        corpus = self._corpus(collections)
        if not corpus.chunks:
            return RetrievalResult([], None, False)

        semantic = cosine_scores(corpus.matrix, self.embedder.embed([query])[0])
        keyword = corpus.keywords.scores(query)
        top_score = float(semantic.max())
        if top_score < min_score:
            return RetrievalResult([], top_score, False)

        ranked = [
            replace(corpus.chunks[i], score=float(semantic[i]))
            for i in _fuse(semantic, keyword)[:top_k]
        ]
        if max_context_chars is not None:
            ranked = _limit_length(ranked, max_context_chars)
        return RetrievalResult(ranked, top_score, True)

    def _corpus(self, collections: Sequence[str]) -> _Corpus:
        key = tuple(sorted(collections))
        version = self.store.version()
        cached = self._cache.get(key)
        if cached is not None and cached[0] == version:
            return cached[1]
        chunks, matrix = self.store.load_chunks(key)
        keywords = KeywordIndex([f"{c.heading or ''}\n{c.text}" for c in chunks])
        corpus = _Corpus(chunks, matrix, keywords)
        self._cache[key] = (version, corpus)
        return corpus


def _fuse(semantic: np.ndarray, keyword: np.ndarray) -> list[int]:
    """語意與關鍵字各自排名後以 RRF 合併；關鍵字分數為 0 的段落不計關鍵字排名。"""
    fused = np.zeros(len(semantic), dtype=np.float64)
    for rank, i in enumerate(np.argsort(-semantic, kind="stable")):
        fused[i] += 1 / (RRF_K + rank + 1)
    for rank, i in enumerate(np.argsort(-keyword, kind="stable")):
        if keyword[i] <= 0:
            break
        fused[i] += 1 / (RRF_K + rank + 1)
    return [int(i) for i in np.argsort(-fused, kind="stable")]


def _limit_length(sources: list[Source], max_chars: int) -> list[Source]:
    """依排名加入段落，總字數不超過 max_chars；放不下的略過。第 1 段超過上限時截斷。"""
    if not sources:
        return []
    first = sources[0]
    if len(first.text) > max_chars:
        return [replace(first, text=first.text[: max_chars - len(TRUNCATED_MARK)] + TRUNCATED_MARK)]
    selected, total = [first], len(first.text)
    for s in sources[1:]:
        if total + len(s.text) <= max_chars:
            selected.append(s)
            total += len(s.text)
    return selected

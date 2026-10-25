"""Embedder 介面與 Ollama 實作（設計見 docs/02-design.md §4.1、§5）。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import PurePath
from typing import Any, Protocol

import numpy as np
import ollama

from rag_chatbot.config import Config
from rag_chatbot.ollama_errors import translate_errors

# progress(已完成數, 總數)
ProgressCallback = Callable[[int, int], None]


class EmbedderError(Exception):
    """向量化失敗（Ollama 未啟動、模型未下載等）。"""


class Embedder(Protocol):
    model: str

    def embed(self, texts: Sequence[str], progress: ProgressCallback | None = None) -> np.ndarray:
        """回傳 float32 陣列，形狀為 (len(texts), 維度)。"""
        ...


def embedding_text(filename: str, heading: str | None, text: str) -> str:
    """向量化用的文字：加上「檔名 > 標題」前綴（D-018），讓只出現在標題的詞也能被檢索到。"""
    prefix = " > ".join(p for p in (PurePath(filename).stem, heading) if p)
    return f"{prefix}\n{text}"


class OllamaEmbedder:
    def __init__(
        self,
        host: str,
        model: str,
        timeout_seconds: float,
        batch_size: int = 16,
        client: Any = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size 必須大於 0")
        self.host = host
        self.model = model
        self.batch_size = batch_size
        self._client = client or ollama.Client(host=host, timeout=timeout_seconds)

    @classmethod
    def from_config(cls, config: Config) -> OllamaEmbedder:
        o = config.ollama
        return cls(o.host, o.embed_model, o.timeout_seconds)

    def embed(self, texts: Sequence[str], progress: ProgressCallback | None = None) -> np.ndarray:
        texts = list(texts)
        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.append(self._embed_batch(batch))
            if progress:
                progress(start + len(batch), len(texts))
        if not vectors:
            return np.zeros((0, 0), dtype=np.float32)
        return np.vstack(vectors)

    def _embed_batch(self, batch: list[str]) -> np.ndarray:
        with translate_errors(EmbedderError, self.host, self.model, "向量化"):
            response = self._client.embed(model=self.model, input=batch)

        result = np.asarray(response.embeddings, dtype=np.float32)
        if result.ndim != 2 or result.shape[0] != len(batch):
            raise EmbedderError(
                f"Ollama 回傳的向量數量不符（送出 {len(batch)} 筆，收到 {len(response.embeddings)} 筆）"
            )
        return result

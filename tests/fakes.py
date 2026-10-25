"""單元測試用的假元件，不需要 Ollama（docs/04-testing.md §1）。"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from rag_chatbot.embedder import EmbedderError, ProgressCallback
from rag_chatbot.llm import LLMError, Message

DEFAULT_KEYWORDS = (
    "保固", "濾網", "方案", "進階版", "價格",
    "履歷", "工作", "專長", "小算盤", "共用",
    "備份", "部落格", "天氣",
)


class FakeEmbedder:
    """以關鍵字出現次數產生固定向量。

    每個關鍵字佔一維，最後一維為常數，避免沒有關鍵字的文字變成零向量。
    含相同關鍵字的文字相似度高，沒有共同關鍵字的相似度低。
    """

    def __init__(
        self,
        keywords: Sequence[str] = DEFAULT_KEYWORDS,
        model: str = "fake-embed",
        fail_on_call: int | None = None,
    ) -> None:
        self.keywords = tuple(keywords)
        self.model = model
        self.fail_on_call = fail_on_call  # 第 N 次呼叫時丟出錯誤（從 1 起算），模擬 Ollama 中途失敗
        self.calls: list[list[str]] = []

    @property
    def dim(self) -> int:
        return len(self.keywords) + 1

    def embed(self, texts: Sequence[str], progress: ProgressCallback | None = None) -> np.ndarray:
        self.calls.append(list(texts))
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise EmbedderError("FakeEmbedder 模擬失敗")
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for j, keyword in enumerate(self.keywords):
                vectors[i, j] = text.count(keyword)
            vectors[i, -1] = 0.1
        if progress:
            progress(len(texts), len(texts))
        return vectors


class FakeLLM:
    """回傳固定文字，或依訊息內容產生回答；記錄每次呼叫的訊息。"""

    def __init__(
        self,
        reply: str | Callable[[Sequence[Message]], str] = "這是測試回答 [1]。",
        model: str = "fake-llm",
        fail: bool = False,
    ) -> None:
        self.reply = reply
        self.model = model
        self.fail = fail
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: Sequence[Message]) -> str:
        self.calls.append([dict(m) for m in messages])
        if self.fail:
            raise LLMError("FakeLLM 模擬失敗")
        return self.reply(messages) if callable(self.reply) else self.reply

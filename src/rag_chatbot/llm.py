"""LLM 介面與 Ollama 實作（設計見 docs/02-design.md §4.2 步驟 8）。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import ollama

from rag_chatbot.config import Config
from rag_chatbot.ollama_errors import translate_errors

# 訊息格式：{"role": "system" | "user" | "assistant", "content": "..."}
Message = Mapping[str, str]


class LLMError(Exception):
    """LLM 呼叫失敗（Ollama 未啟動、模型未下載等）。"""


class LLM(Protocol):
    model: str

    def chat(self, messages: Sequence[Message]) -> str:
        """回傳模型的回答文字。"""
        ...


class OllamaLLM:
    def __init__(
        self,
        host: str,
        model: str,
        timeout_seconds: float,
        num_ctx: int,
        max_tokens: int,
        temperature: float,
        keep_alive: str,
        client: Any = None,
    ) -> None:
        self.host = host
        self.model = model
        self.keep_alive = keep_alive
        self.options = {"num_ctx": num_ctx, "num_predict": max_tokens, "temperature": temperature}
        self._client = client or ollama.Client(host=host, timeout=timeout_seconds)

    @classmethod
    def from_config(cls, config: Config) -> OllamaLLM:
        o = config.ollama
        return cls(
            o.host, o.chat_model, o.timeout_seconds,
            o.chat_num_ctx, o.chat_max_tokens, o.chat_temperature, o.keep_alive,
        )

    def chat(self, messages: Sequence[Message]) -> str:
        with translate_errors(LLMError, self.host, self.model, "回答"):
            response = self._client.chat(
                model=self.model,
                messages=[dict(m) for m in messages],
                options=self.options,
                keep_alive=self.keep_alive,
            )
        text = (response.message.content or "").strip()
        if not text:
            raise LLMError("Ollama 回傳空白回答")
        return text

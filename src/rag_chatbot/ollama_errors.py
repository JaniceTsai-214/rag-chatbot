"""將 Ollama client 的例外轉成本專案的錯誤訊息（Embedder、LLM 共用）。"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import httpx
import ollama


@contextmanager
def translate_errors(error_cls: type[Exception], host: str, model: str, action: str) -> Iterator[None]:
    """action 為動作名稱（如「向量化」「回答」），用於錯誤訊息。"""
    try:
        yield
    except ConnectionError as e:
        raise error_cls(f"無法連線到 Ollama（{host}），請確認 Ollama 已啟動") from e
    except httpx.TimeoutException as e:
        raise error_cls(f"Ollama 回應逾時（{host}）") from e
    except ollama.ResponseError as e:
        if e.status_code == 404:
            raise error_cls(f"Ollama 找不到模型 {model}，請先執行：ollama pull {model}") from e
        raise error_cls(f"Ollama {action}失敗：{e.error}") from e

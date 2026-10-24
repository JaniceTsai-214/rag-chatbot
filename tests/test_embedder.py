from types import SimpleNamespace

import httpx
import numpy as np
import ollama
import pytest

from rag_chatbot.config import load_config
from rag_chatbot.embedder import EmbedderError, OllamaEmbedder, embedding_text
from tests.conftest import PROJECT_ROOT


class FakeOllamaClient:
    """模擬 ollama.Client.embed：回傳每筆文字長度組成的向量，或丟出指定錯誤。"""

    def __init__(self, error: Exception | None = None, drop_one: bool = False):
        self.error = error
        self.drop_one = drop_one
        self.batches: list[list[str]] = []

    def embed(self, model, input):
        self.batches.append(list(input))
        if self.error:
            raise self.error
        vectors = [[float(len(t)), 1.0] for t in input]
        return SimpleNamespace(embeddings=vectors[:-1] if self.drop_one else vectors)


def _embedder(client, batch_size=2):
    return OllamaEmbedder("http://host:1", "bge-m3", 10, batch_size=batch_size, client=client)


# ---------- 前綴 ----------

@pytest.mark.parametrize(
    "filename, heading, expected",
    [
        ("王小明履歷.docx", "工作經歷 > 小算盤團隊", "王小明履歷 > 工作經歷 > 小算盤團隊\n內容"),
        ("手冊.pdf", None, "手冊\n內容"),
        ("faq.v2.csv", "功能", "faq.v2 > 功能\n內容"),
    ],
)
def test_embedding_text(filename, heading, expected):
    assert embedding_text(filename, heading, "內容") == expected


# ---------- 批次與進度 ----------

def test_batches_and_progress():
    client = FakeOllamaClient()
    progress = []
    vectors = _embedder(client).embed(["a", "bb", "ccc", "dddd", "e"], lambda done, total: progress.append((done, total)))
    assert client.batches == [["a", "bb"], ["ccc", "dddd"], ["e"]]
    assert progress == [(2, 5), (4, 5), (5, 5)]
    assert vectors.dtype == np.float32
    assert vectors[:, 0].tolist() == [1, 2, 3, 4, 1]


def test_empty_input():
    client = FakeOllamaClient()
    vectors = _embedder(client).embed([])
    assert vectors.shape == (0, 0)
    assert client.batches == []


def test_invalid_batch_size():
    with pytest.raises(ValueError):
        _embedder(FakeOllamaClient(), batch_size=0)


def test_from_config():
    embedder = OllamaEmbedder.from_config(load_config(PROJECT_ROOT / "config.toml"))
    assert embedder.model == "bge-m3"
    assert embedder.host == "http://127.0.0.1:11434"


# ---------- 錯誤訊息 ----------

@pytest.mark.parametrize(
    "error, message",
    [
        (ConnectionError("refused"), "無法連線到 Ollama（http://host:1）"),
        (httpx.ReadTimeout("slow"), "逾時"),
        (ollama.ResponseError("model 'bge-m3' not found", 404), "ollama pull bge-m3"),
        (ollama.ResponseError("boom", 500), "向量化失敗：boom"),
    ],
)
def test_errors(error, message):
    with pytest.raises(EmbedderError, match=message):
        _embedder(FakeOllamaClient(error=error)).embed(["a"])


def test_count_mismatch():
    with pytest.raises(EmbedderError, match="數量不符"):
        _embedder(FakeOllamaClient(drop_one=True)).embed(["a", "b"])


# ---------- 真實 Ollama（pytest -m ollama） ----------

@pytest.mark.ollama
def test_real_ollama_bge_m3():
    embedder = OllamaEmbedder.from_config(load_config(PROJECT_ROOT / "config.toml"))
    texts = ["小算盤進階版每月 60 元", "進階版一個月多少錢？", "明天台北天氣如何？"]
    v = embedder.embed(texts)
    assert v.shape == (3, 1024)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    assert v[0] @ v[1] > v[0] @ v[2]

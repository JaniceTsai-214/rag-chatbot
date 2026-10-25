from types import SimpleNamespace

import httpx
import ollama
import pytest

from rag_chatbot.config import load_config
from rag_chatbot.llm import LLMError, OllamaLLM
from tests.conftest import PROJECT_ROOT
from tests.fakes import FakeLLM


class FakeOllamaClient:
    def __init__(self, content="回答", error=None):
        self.content = content
        self.error = error
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(message=SimpleNamespace(content=self.content))


def _llm(client):
    return OllamaLLM("http://host:1", "qwen2.5:3b", 10, 4096, 300, 0.2, "30m", client=client)


def test_chat_passes_options_and_strips():
    client = FakeOllamaClient(content="  回答內容 \n")
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    assert _llm(client).chat(messages) == "回答內容"
    call = client.calls[0]
    assert call["model"] == "qwen2.5:3b"
    assert call["messages"] == messages
    assert call["options"] == {"num_ctx": 4096, "num_predict": 300, "temperature": 0.2}
    assert call["keep_alive"] == "30m"


@pytest.mark.parametrize("content", ["", "   ", None])
def test_empty_answer(content):
    with pytest.raises(LLMError, match="空白回答"):
        _llm(FakeOllamaClient(content=content)).chat([{"role": "user", "content": "u"}])


@pytest.mark.parametrize(
    "error, message",
    [
        (ConnectionError("refused"), "無法連線到 Ollama"),
        (httpx.ReadTimeout("slow"), "逾時"),
        (ollama.ResponseError("not found", 404), "ollama pull qwen2.5:3b"),
        (ollama.ResponseError("boom", 500), "回答失敗：boom"),
    ],
)
def test_errors(error, message):
    with pytest.raises(LLMError, match=message):
        _llm(FakeOllamaClient(error=error)).chat([{"role": "user", "content": "u"}])


def test_from_config():
    llm = OllamaLLM.from_config(load_config(PROJECT_ROOT / "config.toml"))
    assert llm.model == "qwen2.5:3b"
    assert llm.keep_alive == "30m"
    assert llm.options == {"num_ctx": 4096, "num_predict": 300, "temperature": 0.2}


def test_fake_llm():
    fixed = FakeLLM()
    assert fixed.chat([{"role": "user", "content": "q"}]) == "這是測試回答 [1]。"
    assert fixed.calls == [[{"role": "user", "content": "q"}]]
    echo = FakeLLM(reply=lambda messages: messages[-1]["content"].upper())
    assert echo.chat([{"role": "user", "content": "abc"}]) == "ABC"
    with pytest.raises(LLMError):
        FakeLLM(fail=True).chat([])


@pytest.mark.ollama
def test_real_ollama_chat():
    llm = OllamaLLM.from_config(load_config(PROJECT_ROOT / "config.toml"))
    text = llm.chat([
        {"role": "system", "content": "只用一個詞回答。"},
        {"role": "user", "content": "日本的首都是哪個城市？"},
    ])
    assert "東京" in text

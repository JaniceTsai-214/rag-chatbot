from pathlib import Path

import pytest

from rag_chatbot.config import Config, ConfigError, load_config
from tests.conftest import PROJECT_ROOT


def test_load_project_config():
    config = load_config(PROJECT_ROOT / "config.toml")
    assert config.default_profile == "intro"
    assert config.ollama.chat_model == "qwen2.5:3b"
    assert config.ollama.embed_model == "bge-m3"
    assert config.chunking.chunk_size == 600
    assert config.retrieval.min_score == 0.40
    assert config.paths.collections == PROJECT_ROOT / "data" / "collections"


def test_empty_file_uses_defaults(write_file):
    path = write_file("config.toml", "")
    config = load_config(path)
    defaults = Config()
    assert config.ollama == defaults.ollama
    assert config.chunking == defaults.chunking
    assert config.retrieval == defaults.retrieval


def test_paths_relative_to_config_file(write_file, tmp_path: Path):
    path = write_file("sub/config.toml", '[paths]\ndb = "x/app.db"\nprofiles = "p"\n')
    config = load_config(path)
    assert config.paths.db == (tmp_path / "sub" / "x" / "app.db").resolve()
    assert config.paths.profiles == (tmp_path / "sub" / "p").resolve()
    assert config.paths.collections == (tmp_path / "sub" / "data" / "collections").resolve()


def test_int_accepted_for_float(write_file):
    config = load_config(write_file("c.toml", "[retrieval]\nmin_score = 1\n"))
    assert config.retrieval.min_score == 1.0
    assert isinstance(config.retrieval.min_score, float)


def test_missing_file(tmp_path: Path):
    with pytest.raises(ConfigError, match="找不到設定檔"):
        load_config(tmp_path / "nope.toml")


@pytest.mark.parametrize(
    "content, message",
    [
        ("[ollama\n", "格式錯誤"),
        ("[unknown]\n", "未知的項目"),
        ("[ollama]\nmodel = 'x'\n", r"\[ollama\] 有未知的項目"),
        ("ollama = 1\n", "必須是區塊"),
        ("[retrieval]\ntop_k = '5'\n", "型別錯誤"),
        ("[retrieval]\ntop_k = true\n", "型別錯誤"),
        ("[paths]\ndb = 1\n", "字串路徑"),
        ("default_profile = ''\n", "default_profile"),
        ("[chunking]\nchunk_size = 0\n", "chunk_size"),
        ("[chunking]\nchunk_size = 100\nchunk_overlap = 100\n", "chunk_overlap"),
        ("[retrieval]\ntop_k = 0\n", "top_k"),
        ("[retrieval]\nmin_score = 1.5\n", "min_score"),
        ("[retrieval]\nhistory_turns = -1\n", "history_turns"),
        ("[ollama]\ntimeout_seconds = 0\n", "timeout_seconds"),
    ],
)
def test_invalid_config(write_file, content, message):
    with pytest.raises(ConfigError, match=message):
        load_config(write_file("c.toml", content))

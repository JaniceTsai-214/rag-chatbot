from pathlib import Path

import pytest

from rag_chatbot.config import load_config
from rag_chatbot.profiles import ProfileError, list_profiles, load_profile
from tests.conftest import PROJECT_ROOT

VALID = """
title = "測試助理"
subject = "王小明"
collections = ["me", "product"]
contact = "test@example.com"
refusal_text = "沒有資料，請洽 {contact}。"
system_prompt = "你是 {subject} 的助理。找不到時回答：{refusal_text} 保留 {other}"
"""


@pytest.fixture
def config(write_file):
    path = write_file("config.toml", "")
    return load_config(path)


@pytest.fixture
def write_profile(write_file):
    def _write(name: str, content: str) -> Path:
        return write_file(f"profiles/{name}.toml", content)

    return _write


def test_project_profiles_are_valid():
    config = load_config(PROJECT_ROOT / "config.toml")
    profiles = {p.name: p for p in list_profiles(config)}
    assert set(profiles) == {"intro", "me", "product-a"}
    assert profiles["intro"].collections == ("me", "product-a")
    assert "tsaichiahui@gmail.com" in profiles["intro"].refusal_text
    assert "{" not in profiles["intro"].render_prompt()
    assert load_profile(config).name == config.default_profile


def test_load_and_substitute(config, write_profile):
    write_profile("demo", VALID)
    profile = load_profile(config, "demo")
    assert profile.title == "測試助理"
    assert profile.collections == ("me", "product")
    assert profile.refusal_text == "沒有資料，請洽 test@example.com。"
    assert profile.render_prompt() == (
        "你是 王小明 的助理。找不到時回答：沒有資料，請洽 test@example.com。 保留 {other}"
    )
    assert profile.render_prompt("{subject}／{contact}") == "王小明／test@example.com"


def test_retrieval_overrides(config, write_profile):
    write_profile("plain", VALID)
    write_profile("tuned", VALID + "top_k = 3\nmin_score = 1\n")
    plain = load_profile(config, "plain")
    tuned = load_profile(config, "tuned")
    assert plain.resolved_top_k(config) == config.retrieval.top_k
    assert plain.resolved_min_score(config) == config.retrieval.min_score
    assert tuned.resolved_top_k(config) == 3
    assert tuned.resolved_min_score(config) == 1.0


def test_missing_collections(config, write_profile, tmp_path):
    write_profile("demo", VALID)
    (tmp_path / "data" / "collections" / "me").mkdir(parents=True)
    assert load_profile(config, "demo").missing_collections(config) == ["product"]


def test_list_profiles_sorted(config, write_profile):
    write_profile("b", VALID)
    write_profile("a", VALID)
    assert [p.name for p in list_profiles(config)] == ["a", "b"]


def test_list_profiles_missing_folder(config):
    with pytest.raises(ProfileError, match="找不到情境資料夾"):
        list_profiles(config)


def test_profile_not_found(config, write_profile):
    write_profile("demo", VALID)
    with pytest.raises(ProfileError, match="找不到情境「nope」"):
        load_profile(config, "nope")


@pytest.mark.parametrize(
    "content, message",
    [
        ("title = \n", "格式錯誤"),
        ('title = "x"\n', "缺少項目"),
        (VALID + "extra = 1\n", "未知的項目"),
        (VALID.replace('subject = "王小明"', 'subject = "  "'), "subject 必須是非空字串"),
        (VALID.replace('["me", "product"]', "[]"), "collections"),
        (VALID.replace('["me", "product"]', '"me"'), "collections"),
        (VALID.replace('["me", "product"]', '["../x"]'), "不合法"),
        (VALID.replace('["me", "product"]', '[".."]'), "不合法"),
        (VALID + "top_k = 0\n", "top_k"),
        (VALID + "top_k = true\n", "top_k"),
        (VALID + "min_score = 'x'\n", "min_score 必須是數字"),
        (VALID + "min_score = 2.0\n", "介於 0 與 1"),
    ],
)
def test_invalid_profile(config, write_profile, content, message):
    write_profile("bad", content)
    with pytest.raises(ProfileError, match=message):
        load_profile(config, "bad")

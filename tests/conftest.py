"""共用 fixture。"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_chatbot.config import Config, load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROFILE_TEMPLATE = """
title = "{title}"
subject = "王小明"
collections = {collections}
contact = "test@example.com"
refusal_text = "沒有資料，請洽 {{contact}}。"
system_prompt = "你是 {{subject}} 的助理。"
"""


@pytest.fixture
def write_file(tmp_path: Path):
    """在 tmp_path 下寫入 UTF-8 文字檔，回傳路徑。"""

    def _write(relative: str, content: str) -> Path:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    return _write


@pytest.fixture
def project(tmp_path: Path, write_file) -> Config:
    """暫存專案：預設設定、intro（me + product）與 me 兩個情境、兩個空的文件集資料夾。"""
    write_file("config.toml", "")
    write_file("profiles/intro.toml", PROFILE_TEMPLATE.format(title="介紹助理", collections='["me", "product"]'))
    write_file("profiles/me.toml", PROFILE_TEMPLATE.format(title="個人介紹", collections='["me"]'))
    (tmp_path / "data" / "collections" / "me").mkdir(parents=True)
    (tmp_path / "data" / "collections" / "product").mkdir(parents=True)
    return load_config(tmp_path / "config.toml")

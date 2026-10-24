"""各 Loader 共用的工具。"""

from __future__ import annotations

import re
from pathlib import Path

_TEXT_ENCODINGS = ("utf-8-sig", "cp950")  # Excel 另存的 CSV 常為 Big5（cp950）


class LoaderError(Exception):
    """文件無法解析。"""


def clean_text(text: str) -> str:
    """去除每行行尾空白，連續空行縮成一行，去除前後空白。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def read_text(path: Path) -> str:
    """依序嘗試 UTF-8（含 BOM）、Big5 讀取文字檔。"""
    data = path.read_bytes()
    for encoding in _TEXT_ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise LoaderError(f"{path.name}：無法辨識文字編碼（支援 UTF-8、Big5）")


class HeadingPath:
    """依標題層級維護「上層 > 下層」的標題路徑。"""

    def __init__(self) -> None:
        self._stack: list[tuple[int, str]] = []

    def push(self, level: int, title: str) -> None:
        while self._stack and self._stack[-1][0] >= level:
            self._stack.pop()
        self._stack.append((level, title))

    @property
    def current(self) -> str | None:
        return " > ".join(title for _, title in self._stack) or None

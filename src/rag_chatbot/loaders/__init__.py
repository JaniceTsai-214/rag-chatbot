"""依副檔名解析文件，輸出含位置資訊的文字區塊（設計見 docs/02-design.md §4.1）。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from rag_chatbot.loaders.common import LoaderError
from rag_chatbot.loaders.docx import load_docx
from rag_chatbot.loaders.faq import load_faq
from rag_chatbot.loaders.markdown import load_markdown
from rag_chatbot.loaders.pdf import load_pdf
from rag_chatbot.models import Section

__all__ = ["LoaderError", "SUPPORTED_EXTENSIONS", "file_type", "load"]

# 副檔名 → (文件類型, 讀取函式)
_LOADERS: dict[str, tuple[str, Callable[[Path], list[Section]]]] = {
    ".pdf": ("pdf", load_pdf),
    ".docx": ("docx", load_docx),
    ".md": ("markdown", load_markdown),
    ".csv": ("faq", load_faq),
    ".xlsx": ("faq", load_faq),
}

SUPPORTED_EXTENSIONS = tuple(_LOADERS)


def file_type(path: Path) -> str | None:
    """回傳文件類型（pdf / docx / markdown / faq）；不支援時回傳 None。"""
    entry = _LOADERS.get(path.suffix.lower())
    return entry[0] if entry else None


def load(path: Path) -> list[Section]:
    """解析文件。不支援的格式、空檔、壞檔都丟出 LoaderError。"""
    entry = _LOADERS.get(path.suffix.lower())
    if entry is None:
        raise LoaderError(f"{path.name}：不支援的格式（支援 {', '.join(SUPPORTED_EXTENSIONS)}）")
    if not path.is_file():
        raise LoaderError(f"{path.name}：找不到檔案")
    sections = entry[1](path)
    if not sections:
        raise LoaderError(f"{path.name}：沒有可擷取的文字")
    return sections

"""DOCX：依標題樣式分節，記錄標題路徑；表格每列轉成「欄 | 欄」。"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import docx
from docx.opc.exceptions import PackageNotFoundError
from docx.table import Table
from docx.text.paragraph import Paragraph

from rag_chatbot.loaders.common import HeadingPath, LoaderError, clean_text
from rag_chatbot.models import Section

_HEADING_STYLE = re.compile(r"^(?:Heading|標題)\s*(\d)$", re.IGNORECASE)


def load_docx(path: Path) -> list[Section]:
    try:
        document = docx.Document(str(path))
    except (PackageNotFoundError, zipfile.BadZipFile, KeyError, ValueError) as e:
        raise LoaderError(f"{path.name}：無法開啟 Word 檔（{e}）") from e

    sections: list[Section] = []
    headings = HeadingPath()
    body: list[str] = []

    def flush() -> None:
        text = clean_text("\n".join(body))
        if text:
            sections.append(Section(text=text, heading=headings.current))
        body.clear()

    for block in document.iter_inner_content():
        if isinstance(block, Paragraph):
            level = _heading_level(block)
            if level is not None and block.text.strip():
                flush()
                headings.push(level, block.text.strip())
            else:
                body.append(block.text)
        elif isinstance(block, Table):
            body.extend(_table_rows(block))
    flush()
    return sections


def _heading_level(paragraph: Paragraph) -> int | None:
    name = paragraph.style.name if paragraph.style is not None else ""
    if name == "Title":
        return 0
    match = _HEADING_STYLE.match(name)
    return int(match.group(1)) if match else None


def _table_rows(table: Table) -> list[str]:
    rows = []
    for row in table.rows:
        cells: list[str] = []
        for cell in row.cells:
            text = " ".join(cell.text.split())
            # 合併儲存格會重複出現，只保留一次
            if not cells or cells[-1] != text:
                cells.append(text)
        line = " | ".join(cells).strip(" |")
        if line:
            rows.append(line)
    return rows

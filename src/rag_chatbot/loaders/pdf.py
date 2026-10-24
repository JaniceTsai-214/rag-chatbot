"""PDF：逐頁擷取文字，記錄頁碼。只處理含文字層的 PDF（不做 OCR）。"""

from __future__ import annotations

from pathlib import Path

import pymupdf

from rag_chatbot.loaders.common import LoaderError, clean_text
from rag_chatbot.models import Section


def load_pdf(path: Path) -> list[Section]:
    try:
        doc = pymupdf.open(path)
    except Exception as e:  # PyMuPDF 對壞檔丟出多種例外
        raise LoaderError(f"{path.name}：無法開啟 PDF（{e}）") from e

    with doc:
        if doc.needs_pass:
            raise LoaderError(f"{path.name}：PDF 有密碼保護")
        sections = []
        for number, page in enumerate(doc, start=1):
            text = clean_text(page.get_text())
            if text:
                sections.append(Section(text=text, page=number))
        if not sections and doc.page_count > 0:
            raise LoaderError(f"{path.name}：PDF 沒有文字層（掃描版 PDF 不支援）")
    return sections

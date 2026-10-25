"""FAQ（CSV / XLSX）：每列一筆，內容為「問：… 答：…」，分類記為標題。"""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import openpyxl

from rag_chatbot.loaders.common import LoaderError, read_text
from rag_chatbot.models import Section

# 欄位名稱（不分大小寫）→ 標準欄位
_COLUMNS = {
    "question": "question", "問題": "question",
    "answer": "answer", "答案": "answer", "回答": "answer",
    "category": "category", "分類": "category",
}


def load_faq(path: Path) -> list[Section]:
    rows = _read_xlsx(path) if path.suffix.lower() == ".xlsx" else _read_csv(path)
    rows = [r for r in rows if any(r)]
    if not rows:
        return []

    columns = _map_columns(path, rows[0])
    sections = []
    for number, row in enumerate(rows[1:], start=2):
        values = {key: row[i].strip() if i < len(row) else "" for key, i in columns.items()}
        question, answer = values["question"], values["answer"]
        if not question and not answer:
            continue
        if not question or not answer:
            raise LoaderError(f"{path.name}：第 {number} 列缺少問題或答案")
        sections.append(
            Section(text=f"問：{question}\n答：{answer}", heading=values.get("category") or None)
        )
    return sections


def _read_csv(path: Path) -> list[list[str]]:
    return list(csv.reader(io.StringIO(read_text(path))))


def _read_xlsx(path: Path) -> list[list[str]]:
    try:
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    except (zipfile.BadZipFile, KeyError, ValueError, OSError) as e:
        raise LoaderError(f"{path.name}：無法開啟 Excel 檔（{e}）") from e
    try:
        sheet = workbook.worksheets[0]
        return [["" if v is None else str(v) for v in row] for row in sheet.iter_rows(values_only=True)]
    finally:
        workbook.close()


def _map_columns(path: Path, header: list[str]) -> dict[str, int]:
    columns: dict[str, int] = {}
    for i, name in enumerate(header):
        key = _COLUMNS.get(name.strip().lower())
        if key and key not in columns:
            columns[key] = i
    missing = [k for k in ("question", "answer") if k not in columns]
    if missing:
        raise LoaderError(
            f"{path.name}：缺少欄位 {', '.join(missing)}（第一列需有 question / answer 或 問題 / 答案）"
        )
    return columns

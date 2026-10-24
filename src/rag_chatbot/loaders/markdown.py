"""Markdown：依 # 標題分節，記錄標題路徑（如「作品集 > 小算盤」）。"""

from __future__ import annotations

import re
from pathlib import Path

from rag_chatbot.loaders.common import HeadingPath, clean_text, read_text
from rag_chatbot.models import Section

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


def load_markdown(path: Path) -> list[Section]:
    sections: list[Section] = []
    headings = HeadingPath()
    body: list[str] = []
    fence: str | None = None

    def flush() -> None:
        text = clean_text("\n".join(body))
        if text:
            sections.append(Section(text=text, heading=headings.current))
        body.clear()

    for line in read_text(path).splitlines():
        fence_match = _FENCE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = marker
            elif marker == fence:
                fence = None
            body.append(line)
            continue
        heading_match = _HEADING.match(line) if fence is None else None
        if heading_match:
            flush()
            headings.push(len(heading_match.group(1)), heading_match.group(2))
        else:
            body.append(line)
    flush()
    return sections

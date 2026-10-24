"""共用 fixture。"""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def write_file(tmp_path: Path):
    """在 tmp_path 下寫入 UTF-8 文字檔，回傳路徑。"""

    def _write(relative: str, content: str) -> Path:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    return _write

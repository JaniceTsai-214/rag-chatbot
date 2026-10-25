"""讀取 .env（金鑰等機密設定，不進 Git）。"""

from __future__ import annotations

import os
from pathlib import Path


def load_env(path: str | Path) -> dict[str, str]:
    """讀取 KEY=VALUE 格式的檔案；# 開頭為註解，值可用引號包住。檔案不存在時回傳空字典。

    已存在的環境變數優先（方便臨時覆寫）。
    """
    path = Path(path)
    values: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            values[key.strip()] = value
    for key in list(values):
        if key in os.environ:
            values[key] = os.environ[key]
    return values

"""Prompt 版本管理：每個情境各自保存版本（設計見 docs/02-design.md §3、§6）。"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime

from rag_chatbot.profiles import Profile

DEFAULT_NOTE = "情境預設 Prompt"


class PromptError(Exception):
    """指定的 Prompt 版本不存在等問題。"""


@dataclass(frozen=True)
class PromptVersion:
    profile: str
    version: int
    content: str  # 範本（含 {subject} 等變數），使用時以 Profile.render_prompt() 代入
    note: str | None
    created_at: str
    is_active: bool


class PromptStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def ensure_default(self, profile: Profile) -> None:
        """情境還沒有任何版本時，把情境設定檔中的 system_prompt 存為 v1 並啟用。"""
        if not self.list(profile.name):
            self._insert(profile.name, 1, profile.system_prompt, DEFAULT_NOTE, active=True)

    def active(self, profile: Profile) -> PromptVersion:
        """目前啟用的版本（必要時先建立 v1）。"""
        self.ensure_default(profile)
        row = self.conn.execute(
            "SELECT * FROM prompts WHERE profile = ? AND is_active = 1", (profile.name,)
        ).fetchone()
        return _version(row)

    def resolve(self, profile: Profile, version: int | None) -> PromptVersion:
        """version 為 None 時回傳啟用版本，否則回傳指定版本（不存在時丟出 PromptError）。"""
        if version is None:
            return self.active(profile)
        self.ensure_default(profile)
        found = self.get(profile.name, version)
        if found is None:
            raise PromptError(f"情境「{profile.name}」沒有 Prompt v{version}")
        return found

    def get(self, profile_name: str, version: int) -> PromptVersion | None:
        row = self.conn.execute(
            "SELECT * FROM prompts WHERE profile = ? AND version = ?", (profile_name, version)
        ).fetchone()
        return _version(row) if row else None

    def list(self, profile_name: str) -> list[PromptVersion]:
        rows = self.conn.execute(
            "SELECT * FROM prompts WHERE profile = ? ORDER BY version", (profile_name,)
        )
        return [_version(r) for r in rows]

    def add(self, profile: Profile, content: str, note: str | None = None, activate: bool = True) -> PromptVersion:
        """新增一個版本（版本號 = 目前最大 + 1）。"""
        if not content.strip():
            raise PromptError("Prompt 內容不可為空白")
        self.ensure_default(profile)
        row = self.conn.execute(
            "SELECT MAX(version) FROM prompts WHERE profile = ?", (profile.name,)
        ).fetchone()
        version = (row[0] or 0) + 1
        self._insert(profile.name, version, content, note, activate)
        return self.get(profile.name, version)

    def activate(self, profile_name: str, version: int) -> None:
        if self.get(profile_name, version) is None:
            raise PromptError(f"情境「{profile_name}」沒有 Prompt v{version}")
        with self.conn:
            self.conn.execute("UPDATE prompts SET is_active = 0 WHERE profile = ?", (profile_name,))
            self.conn.execute(
                "UPDATE prompts SET is_active = 1 WHERE profile = ? AND version = ?", (profile_name, version)
            )

    def _insert(self, profile_name: str, version: int, content: str, note: str | None, active: bool) -> None:
        with self.conn:
            if active:
                self.conn.execute("UPDATE prompts SET is_active = 0 WHERE profile = ?", (profile_name,))
            self.conn.execute(
                "INSERT INTO prompts(profile, version, content, note, created_at, is_active)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (profile_name, version, content, note, _now(), int(active)),
            )


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _version(row: sqlite3.Row) -> PromptVersion:
    return PromptVersion(
        profile=row["profile"],
        version=row["version"],
        content=row["content"],
        note=row["note"],
        created_at=row["created_at"],
        is_active=bool(row["is_active"]),
    )

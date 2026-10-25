"""LINE 訪客與對話的對應：閒置超過設定時間就開始新對話（docs/05-line.md §4.3）。"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta


class LineUsers:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def active_conversation(self, user_id: str, now: datetime, idle_minutes: int) -> int | None:
        """訪客最近的對話；不存在或閒置超過 idle_minutes 時回傳 None（由呼叫端建立新對話）。"""
        row = self.conn.execute(
            "SELECT conversation_id, last_active_at FROM line_users WHERE user_id = ?", (user_id,)
        ).fetchone()
        if row is None or row["conversation_id"] is None:
            return None
        if now - datetime.fromisoformat(row["last_active_at"]) > timedelta(minutes=idle_minutes):
            return None
        return row["conversation_id"]

    def touch(self, user_id: str, conversation_id: int, now: datetime) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO line_users(user_id, conversation_id, last_active_at) VALUES (?, ?, ?)"
                " ON CONFLICT(user_id) DO UPDATE SET conversation_id = excluded.conversation_id,"
                " last_active_at = excluded.last_active_at",
                (user_id, conversation_id, now.isoformat(timespec="seconds")),
            )

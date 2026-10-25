"""對話紀錄（設計見 docs/02-design.md §6）。"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime

from rag_chatbot.models import Source

USER = "user"
ASSISTANT = "assistant"
TITLE_CHARS = 30


class HistoryError(Exception):
    """對話不存在等問題。"""


@dataclass(frozen=True)
class Conversation:
    id: int
    profile: str
    title: str
    created_at: str


@dataclass(frozen=True)
class Message:
    id: int
    conversation_id: int
    role: str
    content: str
    kind: str | None
    prompt_version: int | None
    model: str | None
    sources: list[Source]
    top_score: float | None
    latency_ms: int | None
    created_at: str
    rerun_of: int | None


class HistoryStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def create_conversation(self, profile: str, title: str) -> Conversation:
        title = " ".join(title.split())
        if len(title) > TITLE_CHARS:
            title = title[: TITLE_CHARS - 1] + "…"
        created_at = _now()
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO conversations(profile, title, created_at) VALUES (?, ?, ?)",
                (profile, title or "新對話", created_at),
            )
        return Conversation(cursor.lastrowid, profile, title or "新對話", created_at)

    def get_conversation(self, conversation_id: int) -> Conversation | None:
        row = self.conn.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        return Conversation(row["id"], row["profile"], row["title"], row["created_at"]) if row else None

    def list_conversations(self, profile: str) -> list[Conversation]:
        """指定情境的對話，最新的在前。"""
        rows = self.conn.execute(
            "SELECT * FROM conversations WHERE profile = ? ORDER BY id DESC", (profile,)
        )
        return [Conversation(r["id"], r["profile"], r["title"], r["created_at"]) for r in rows]

    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        *,
        kind: str | None = None,
        prompt_version: int | None = None,
        model: str | None = None,
        sources: Sequence[Source] = (),
        top_score: float | None = None,
        latency_ms: int | None = None,
        rerun_of: int | None = None,
    ) -> Message:
        if role not in (USER, ASSISTANT):
            raise ValueError(f"未知的角色：{role}")
        if self.get_conversation(conversation_id) is None:
            raise HistoryError(f"對話 {conversation_id} 不存在")
        sources_json = json.dumps([asdict(s) for s in sources], ensure_ascii=False) if sources else None
        created_at = _now()
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO messages(conversation_id, role, content, kind, prompt_version, model,"
                " sources_json, top_score, latency_ms, created_at, rerun_of)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (conversation_id, role, content, kind, prompt_version, model,
                 sources_json, top_score, latency_ms, created_at, rerun_of),
            )
        return self.get_message(cursor.lastrowid)

    def get_message(self, message_id: int) -> Message | None:
        row = self.conn.execute("SELECT * FROM messages WHERE id = ?", (message_id,)).fetchone()
        return _message(row) if row else None

    def messages(self, conversation_id: int) -> list[Message]:
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id", (conversation_id,)
        )
        return [_message(r) for r in rows]

    def recent_questions(self, conversation_id: int, turns: int) -> list[str]:
        """最近 turns 則使用者問題（舊到新），供多輪檢索與 Prompt 使用。"""
        if turns <= 0:
            return []
        rows = self.conn.execute(
            "SELECT content FROM messages WHERE conversation_id = ? AND role = ?"
            " ORDER BY id DESC LIMIT ?",
            (conversation_id, USER, turns),
        ).fetchall()
        return [r["content"] for r in reversed(rows)]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _message(row: sqlite3.Row) -> Message:
    sources = [Source(**s) for s in json.loads(row["sources_json"])] if row["sources_json"] else []
    return Message(
        id=row["id"],
        conversation_id=row["conversation_id"],
        role=row["role"],
        content=row["content"],
        kind=row["kind"],
        prompt_version=row["prompt_version"],
        model=row["model"],
        sources=sources,
        top_score=row["top_score"],
        latency_ms=row["latency_ms"],
        created_at=row["created_at"],
        rerun_of=row["rerun_of"],
    )

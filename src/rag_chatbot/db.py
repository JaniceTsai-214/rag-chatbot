"""SQLite 連線與資料表建立（設計見 docs/02-design.md §6）。"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = "1"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY,
    collection  TEXT NOT NULL,
    filename    TEXT NOT NULL,
    file_type   TEXT NOT NULL,
    sha256      TEXT NOT NULL,
    chunk_count INTEGER NOT NULL,
    imported_at TEXT NOT NULL,
    UNIQUE (collection, filename)
);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL,
    page        INTEGER,
    heading     TEXT,
    embedding   BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);

CREATE TABLE IF NOT EXISTS prompts (
    id         INTEGER PRIMARY KEY,
    profile    TEXT NOT NULL,
    version    INTEGER NOT NULL,
    content    TEXT NOT NULL,
    note       TEXT,
    created_at TEXT NOT NULL,
    is_active  INTEGER NOT NULL DEFAULT 0,
    UNIQUE (profile, version)
);

CREATE TABLE IF NOT EXISTS conversations (
    id         INTEGER PRIMARY KEY,
    profile    TEXT NOT NULL,
    title      TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    prompt_version  INTEGER,
    model           TEXT,
    sources_json    TEXT,
    latency_ms      INTEGER,
    created_at      TEXT NOT NULL,
    rerun_of        INTEGER REFERENCES messages(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
"""


class DatabaseError(Exception):
    """資料庫版本不符等問題。"""


def connect(path: str | Path) -> sqlite3.Connection:
    """開啟資料庫（不存在時建立），並確保資料表存在。"""
    path = Path(path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    with conn:
        conn.executescript(_SCHEMA)
        conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)", (SCHEMA_VERSION,))
    version = get_meta(conn, "schema_version")
    if version != SCHEMA_VERSION:
        conn.close()
        raise DatabaseError(f"資料庫版本 {version} 與程式版本 {SCHEMA_VERSION} 不符：{path}")
    return conn


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    """寫入 meta；不自行 commit，由呼叫端的交易決定。"""
    conn.execute(
        "INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )

import sqlite3

import pytest

from rag_chatbot.db import connect, get_meta
from rag_chatbot.history import HistoryError, HistoryStore
from rag_chatbot.models import Source
from rag_chatbot.profiles import load_profile
from rag_chatbot.prompts import DEFAULT_NOTE, PromptError, PromptStore


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "app.db")
    yield c
    c.close()


@pytest.fixture
def profiles(project):
    return load_profile(project, "intro"), load_profile(project, "me")


# ---------- 資料庫升級 ----------

def test_migrate_v1_to_latest(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript("""
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO meta VALUES ('schema_version', '1');
        CREATE TABLE conversations (id INTEGER PRIMARY KEY, profile TEXT NOT NULL,
                                    title TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE messages (id INTEGER PRIMARY KEY, conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL, content TEXT NOT NULL, prompt_version INTEGER, model TEXT,
            sources_json TEXT, latency_ms INTEGER, created_at TEXT NOT NULL, rerun_of INTEGER);
        INSERT INTO conversations VALUES (1, 'intro', '舊對話', '2026-10-24T00:00:00');
        INSERT INTO messages(conversation_id, role, content, created_at) VALUES (1, 'user', '舊問題', 'x');
    """)
    old.commit()
    old.close()

    conn = connect(path)
    assert get_meta(conn, "schema_version") == "3"
    assert conn.execute("SELECT COUNT(*) FROM line_users").fetchone()[0] == 0
    columns = {r[1] for r in conn.execute("PRAGMA table_info(messages)")}
    assert {"kind", "top_score"} <= columns
    message = HistoryStore(conn).messages(1)[0]
    assert (message.content, message.kind, message.top_score) == ("舊問題", None, None)
    conn.close()
    connect(path).close()  # 再次開啟不會重複升級


# ---------- Prompt 版本 ----------

def test_default_prompt_created_once(conn, profiles):
    intro, _ = profiles
    store = PromptStore(conn)
    assert store.list("intro") == []
    active = store.active(intro)
    assert (active.version, active.content, active.note, active.is_active) == (1, intro.system_prompt, DEFAULT_NOTE, True)
    store.active(intro)
    assert len(store.list("intro")) == 1


def test_add_and_activate_versions(conn, profiles):
    intro, me = profiles
    store = PromptStore(conn)
    v2 = store.add(intro, "以英文回答。", note="英文測試")
    assert (v2.version, v2.is_active) == (2, True)
    assert store.active(intro).version == 2
    assert [v.is_active for v in store.list("intro")] == [False, True]

    v3 = store.add(intro, "第三版", activate=False)
    assert v3.is_active is False and store.active(intro).version == 2

    store.activate("intro", 1)
    assert store.active(intro).version == 1
    assert store.resolve(intro, None).version == 1
    assert store.resolve(intro, 3).content == "第三版"

    # 其他情境不受影響
    assert store.active(me).version == 1 and len(store.list("me")) == 1


def test_prompt_errors(conn, profiles):
    intro, _ = profiles
    store = PromptStore(conn)
    with pytest.raises(PromptError, match="不可為空白"):
        store.add(intro, "  ")
    with pytest.raises(PromptError, match="沒有 Prompt v9"):
        store.activate("intro", 9)
    with pytest.raises(PromptError, match="沒有 Prompt v9"):
        store.resolve(intro, 9)
    assert store.get("intro", 9) is None


# ---------- 對話紀錄 ----------

def _source(i, score=0.5):
    return Source(i, "me", "履歷.md", None, "學歷", f"內容{i}", score)


def test_conversation_and_messages(conn):
    history = HistoryStore(conn)
    c = history.create_conversation("intro", "  她的學歷\n是什麼？" + "很長" * 30)
    assert c.title.startswith("她的學歷 是什麼？") and len(c.title) == 30 and c.title.endswith("…")
    assert history.get_conversation(c.id) == c
    assert history.get_conversation(999) is None

    q = history.add_message(c.id, "user", "她的學歷是什麼？")
    a = history.add_message(
        c.id, "assistant", "清華核工 [1]", kind="answered", prompt_version=1, model="qwen2.5:3b",
        sources=[_source(1, 0.61), _source(2)], top_score=0.61, latency_ms=12345,
    )
    assert (q.kind, q.sources, q.top_score) == (None, [], None)
    assert a.sources == [_source(1, 0.61), _source(2)]
    assert (a.kind, a.prompt_version, a.model, a.top_score, a.latency_ms) == ("answered", 1, "qwen2.5:3b", 0.61, 12345)

    rerun = history.add_message(c.id, "assistant", "重跑", kind="answered", prompt_version=2, rerun_of=a.id)
    assert rerun.rerun_of == a.id
    assert [m.id for m in history.messages(c.id)] == [q.id, a.id, rerun.id]


def test_list_conversations_by_profile(conn):
    history = HistoryStore(conn)
    first = history.create_conversation("intro", "一")
    second = history.create_conversation("intro", "")
    history.create_conversation("me", "三")
    assert [c.id for c in history.list_conversations("intro")] == [second.id, first.id]
    assert second.title == "新對話"
    assert len(history.list_conversations("me")) == 1


def test_recent_questions(conn):
    history = HistoryStore(conn)
    c = history.create_conversation("intro", "x")
    assert history.recent_questions(c.id, 2) == []
    for i in range(3):
        history.add_message(c.id, "user", f"問題{i}")
        history.add_message(c.id, "assistant", f"回答{i}")
    assert history.recent_questions(c.id, 2) == ["問題1", "問題2"]
    assert history.recent_questions(c.id, 0) == []


def test_message_errors(conn):
    history = HistoryStore(conn)
    with pytest.raises(HistoryError, match="不存在"):
        history.add_message(999, "user", "x")
    c = history.create_conversation("intro", "x")
    with pytest.raises(ValueError, match="未知的角色"):
        history.add_message(c.id, "system", "x")
    assert history.get_message(999) is None

import sqlite3

import numpy as np
import pytest

from rag_chatbot.db import DatabaseError, connect, get_meta, set_meta
from rag_chatbot.models import Chunk
from rag_chatbot.vector_store import VectorStore, VectorStoreError
from tests.fakes import FakeEmbedder


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "sub" / "app.db"


@pytest.fixture
def store(db_path):
    conn = connect(db_path)
    yield VectorStore(conn, "fake-embed")
    conn.close()


@pytest.fixture
def embedder():
    return FakeEmbedder()


def _add(store, embedder, collection, filename, texts, sha="h1", file_type="markdown"):
    chunks = [Chunk(i, t, page=i + 1) for i, t in enumerate(texts)]
    return store.replace_document(collection, filename, file_type, sha, chunks, embedder.embed(texts))


# ---------- db ----------

def test_connect_creates_file_and_tables(db_path):
    conn = connect(db_path)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"meta", "documents", "chunks", "prompts", "conversations", "messages"} <= tables
    assert get_meta(conn, "schema_version") == "2"
    conn.close()
    assert db_path.is_file()
    connect(db_path).close()  # 再次開啟不會出錯


def test_schema_version_mismatch(db_path):
    conn = connect(db_path)
    with conn:
        set_meta(conn, "schema_version", "99")
    conn.close()
    with pytest.raises(DatabaseError, match="版本 99"):
        connect(db_path)


def test_memory_database():
    conn = connect(":memory:")
    assert get_meta(conn, "missing") is None
    conn.close()


# ---------- 文件 ----------

def test_add_and_list(store, embedder):
    info = _add(store, embedder, "product", "手冊.pdf", ["方案價格", "備份說明"], file_type="pdf")
    assert info.chunk_count == 2
    assert store.get_document("product", "手冊.pdf") == info
    assert store.get_document("product", "nope.pdf") is None
    _add(store, embedder, "me", "履歷.docx", ["工作經歷"])
    _add(store, embedder, "me", "作品集.md", ["部落格"])
    assert [(d.collection, d.filename) for d in store.list_documents()] == [
        ("me", "作品集.md"), ("me", "履歷.docx"), ("product", "手冊.pdf"),
    ]
    assert [d.filename for d in store.list_documents("me")] == ["作品集.md", "履歷.docx"]
    assert store.chunk_count() == 4


def test_replace_document(store, embedder):
    _add(store, embedder, "me", "a.md", ["舊內容一", "舊內容二", "舊內容三"], sha="old")
    info = _add(store, embedder, "me", "a.md", ["新內容"], sha="new")
    assert store.get_document("me", "a.md").sha256 == "new"
    assert info.chunk_count == 1
    assert store.chunk_count() == 1
    texts = [r[0] for r in store.conn.execute("SELECT text FROM chunks")]
    assert texts == ["新內容"]


def test_replace_failure_keeps_old_data(store, embedder):
    _add(store, embedder, "me", "a.md", ["舊內容"], sha="old")
    bad = np.zeros((1, embedder.dim), dtype=np.float32)
    with pytest.raises(VectorStoreError, match="向量為零"):
        store.replace_document("me", "a.md", "markdown", "new", [Chunk(0, "新內容")], bad)
    assert store.get_document("me", "a.md").sha256 == "old"
    assert [r[0] for r in store.conn.execute("SELECT text FROM chunks")] == ["舊內容"]


def test_replace_rolls_back_on_database_error(store, embedder):
    _add(store, embedder, "me", "a.md", ["舊內容"], sha="old")
    chunks = [Chunk(0, "新一"), Chunk(0, None)]  # text 為 NULL 違反 NOT NULL
    with pytest.raises(sqlite3.IntegrityError):
        store.replace_document("me", "a.md", "markdown", "new", chunks, embedder.embed(["保固", "方案"]))
    assert store.get_document("me", "a.md").sha256 == "old"
    assert store.chunk_count() == 1


@pytest.mark.parametrize("texts, rows", [([], 0), (["a", "b"], 1)])
def test_replace_validates_counts(store, embedder, texts, rows):
    chunks = [Chunk(i, t) for i, t in enumerate(texts)]
    vectors = embedder.embed(["x"] * rows) if rows else np.zeros((0, embedder.dim), dtype=np.float32)
    with pytest.raises(VectorStoreError):
        store.replace_document("me", "a.md", "markdown", "h", chunks, vectors)


def test_dimension_mismatch(store, embedder):
    _add(store, embedder, "me", "a.md", ["保固"])
    other = FakeEmbedder(keywords=("保固",))
    with pytest.raises(VectorStoreError, match="維度"):
        _add(store, other, "me", "b.md", ["保固"])
    # 取代唯一一份文件時不受舊維度限制
    _add(store, other, "me", "a.md", ["保固"])


def test_delete_document(store, embedder):
    _add(store, embedder, "me", "a.md", ["一", "二"])
    _add(store, embedder, "me", "b.md", ["三"])
    assert store.delete_document("me", "a.md") is True
    assert store.delete_document("me", "a.md") is False
    assert [d.filename for d in store.list_documents()] == ["b.md"]
    assert store.chunk_count() == 1  # 段落一併刪除


def test_clear(store, embedder):
    _add(store, embedder, "me", "a.md", ["一"])
    _add(store, embedder, "product", "b.md", ["二"])
    assert store.clear() == 2
    assert store.list_documents() == []
    assert store.chunk_count() == 0


def test_embedding_stored_normalized(store, embedder):
    _add(store, embedder, "me", "a.md", ["保固保固保固"])
    blob = store.conn.execute("SELECT embedding FROM chunks").fetchone()[0]
    vector = np.frombuffer(blob, dtype=np.float32)
    assert vector.shape == (embedder.dim,)
    assert np.isclose(np.linalg.norm(vector), 1.0)


# ---------- embedding 模型 ----------

def test_embed_model_mismatch(store, embedder, db_path):
    assert store.stored_embed_model() is None
    _add(store, embedder, "me", "a.md", ["保固"])
    assert store.stored_embed_model() == "fake-embed"
    assert store.embed_model_mismatch() is None

    other = VectorStore(store.conn, "bge-m3")
    assert other.embed_model_mismatch() == "fake-embed"
    with pytest.raises(VectorStoreError, match="重新匯入全部文件"):
        other.search(embedder.embed(["保固"])[0], ["me"], 5)
    with pytest.raises(VectorStoreError, match="重新匯入全部文件"):
        _add(other, embedder, "me", "b.md", ["保固"])

    # 清空後即可用新模型重建
    other.clear()
    assert other.embed_model_mismatch() is None
    _add(other, embedder, "me", "b.md", ["保固"])
    assert other.stored_embed_model() == "bge-m3"


# ---------- 搜尋 ----------

def test_search_ranking_and_collection_filter(store, embedder):
    _add(store, embedder, "product", "手冊.pdf", ["方案與價格", "每日備份", "小算盤簡介"], file_type="pdf")
    _add(store, embedder, "me", "履歷.docx", ["工作經歷與專長"])
    query = embedder.embed(["進階版方案價格"])[0]

    results = store.search(query, ["product", "me"], top_k=2)
    assert len(results) == 2
    assert results[0].text == "方案與價格"
    assert results[0].collection == "product"
    assert results[0].filename == "手冊.pdf"
    assert results[0].page == 1
    assert results[0].score > results[1].score
    assert 0.99 < store.search(embedder.embed(["方案與價格"])[0], ["product"], 1)[0].score <= 1.0001

    only_me = store.search(query, ["me"], top_k=5)
    assert [r.filename for r in only_me] == ["履歷.docx"]


def test_search_returns_all_when_fewer_than_top_k(store, embedder):
    _add(store, embedder, "me", "a.md", ["一", "二"])
    assert len(store.search(embedder.embed(["保固"])[0], ["me"], top_k=10)) == 2


def test_search_empty(store, embedder):
    query = embedder.embed(["保固"])[0]
    assert store.search(query, ["me"], 5) == []
    _add(store, embedder, "me", "a.md", ["保固"])
    assert store.search(query, [], 5) == []
    assert store.search(query, ["nope"], 5) == []


def test_search_validation(store, embedder):
    _add(store, embedder, "me", "a.md", ["保固"])
    with pytest.raises(ValueError):
        store.search(embedder.embed(["保固"])[0], ["me"], 0)
    with pytest.raises(VectorStoreError, match="查詢向量維度"):
        store.search(np.ones(3, dtype=np.float32), ["me"], 5)
    with pytest.raises(VectorStoreError, match="查詢"):
        store.search(np.zeros(embedder.dim, dtype=np.float32), ["me"], 5)

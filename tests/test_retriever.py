import numpy as np
import pytest

from rag_chatbot.db import connect
from rag_chatbot.keyword_index import KeywordIndex, tokenize
from rag_chatbot.models import Chunk
from rag_chatbot.retriever import TRUNCATED_MARK, Retriever, _fuse, _limit_length
from rag_chatbot.vector_store import VectorStore
from tests.fakes import FakeEmbedder


# ---------- 斷詞與 BM25 ----------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("學歷", ["學歷"]),
        ("目前工作", ["目前", "前工", "工作"]),
        ("ERP 與 AI", ["erp", "ai", "與"]),
        ("Python3、C# 程式", ["python3", "c", "程式"]),
        ("（）！", []),
    ],
)
def test_tokenize(text, expected):
    assert sorted(tokenize(text)) == sorted(expected)


def test_bm25_ranking():
    index = KeywordIndex(["學歷：清華大學核子工程", "工作經驗：營運長", "學歷與學歷證明", ""])
    scores = index.scores("她的學歷")
    assert scores[1] == 0 and scores[3] == 0
    assert scores[2] > scores[0] > 0  # 出現次數多者較高
    assert not index.scores("完全無關的詞彙").any()


def test_bm25_rare_term_weighs_more():
    index = KeywordIndex(["南亞 排程", "長庚 排程", "新漢 排程"])
    scores = index.scores("南亞排程")
    assert scores[0] > scores[1] == scores[2] > 0


def test_bm25_empty_index():
    assert KeywordIndex([]).scores("任何").shape == (0,)


# ---------- 合併與長度上限 ----------

def test_fuse_combines_rankings():
    semantic = np.array([0.9, 0.8, 0.1])
    keyword = np.array([0.0, 5.0, 3.0])
    # 0：1/61；1：1/62 + 1/61；2：1/63 + 1/62 → 兩邊都有排名的段落勝過只有一邊第一名的
    assert _fuse(semantic, keyword) == [1, 2, 0]
    assert _fuse(semantic, np.zeros(3)) == [0, 1, 2]


def _source(text, i=0):
    from rag_chatbot.models import Source

    return Source(i, "me", "a.md", None, None, text, 0.5)


def test_limit_length_skips_what_does_not_fit():
    sources = [_source("一" * 50, 1), _source("二" * 60, 2), _source("三" * 40, 3)]
    assert [s.chunk_id for s in _limit_length(sources, 100)] == [1, 3]
    assert [s.chunk_id for s in _limit_length(sources, 150)] == [1, 2, 3]
    assert _limit_length([], 100) == []


def test_limit_length_truncates_first():
    result = _limit_length([_source("字" * 200, 1), _source("短", 2)], 100)
    assert len(result) == 1
    assert len(result[0].text) == 100 and result[0].text.endswith(TRUNCATED_MARK)


# ---------- Retriever ----------

@pytest.fixture
def store(tmp_path):
    conn = connect(tmp_path / "app.db")
    yield VectorStore(conn, "fake-embed")
    conn.close()


@pytest.fixture
def embedder():
    return FakeEmbedder()


def _add(store, embedder, collection, filename, items):
    """items：(標題, 內文) 清單。"""
    chunks = [Chunk(i, text, heading=heading) for i, (heading, text) in enumerate(items)]
    store.replace_document(collection, filename, "markdown", filename, chunks,
                           embedder.embed([f"{h}\n{t}" for h, t in items]))


def test_retrieve_hybrid_and_scope(store, embedder):
    _add(store, embedder, "product", "手冊.md", [("方案", "進階版價格每月 60 元"), ("備份", "每日備份")])
    _add(store, embedder, "me", "履歷.md", [("學歷", "清華大學核子工程"), ("工作", "營運長")])
    retriever = Retriever(store, embedder)

    result = retriever.retrieve("進階版價格", ["product", "me"], top_k=3, min_score=0.4)
    assert result.passed
    assert result.sources[0].text == "進階版價格每月 60 元"
    assert result.top_score == pytest.approx(result.sources[0].score)
    assert len(result.sources) == 3

    # FakeEmbedder 不認得「學歷」，語意分數相同；關鍵字檢索把學歷段落排到第 1
    result = retriever.retrieve("她的學歷", ["me"], top_k=2, min_score=0.0)
    assert result.sources[0].heading == "學歷"
    assert all(s.collection == "me" for s in result.sources)


def test_retrieve_threshold(store, embedder):
    _add(store, embedder, "me", "a.md", [("", "保固兩年")])
    result = Retriever(store, embedder).retrieve("天氣如何", ["me"], top_k=5, min_score=0.4)
    assert result.passed is False
    assert result.sources == []
    assert result.top_score is not None and result.top_score < 0.4


def test_retrieve_empty(store, embedder):
    result = Retriever(store, embedder).retrieve("問題", ["me"], top_k=5, min_score=0.4)
    assert result == result.__class__([], None, False)
    with pytest.raises(ValueError):
        Retriever(store, embedder).retrieve("問題", ["me"], top_k=0, min_score=0.4)


def test_retrieve_context_limit(store, embedder):
    _add(store, embedder, "me", "a.md", [("", "保固" + "字" * 98), ("", "保固" + "字" * 98)])
    result = Retriever(store, embedder).retrieve("保固", ["me"], 5, 0.0, max_context_chars=150)
    assert len(result.sources) == 1


def test_cache_rebuilds_after_change(store, embedder):
    retriever = Retriever(store, embedder)
    _add(store, embedder, "me", "a.md", [("", "保固兩年")])
    assert len(retriever.retrieve("保固", ["me"], 5, 0.0).sources) == 1
    calls = len(embedder.calls)
    retriever.retrieve("保固", ["me"], 5, 0.0)
    assert len(embedder.calls) == calls + 1  # 只多了查詢向量，未重建

    _add(store, embedder, "me", "b.md", [("", "保固三年")])
    assert len(retriever.retrieve("保固", ["me"], 5, 0.0).sources) == 2
    store.delete_document("me", "a.md")
    assert [s.text for s in retriever.retrieve("保固", ["me"], 5, 0.0).sources] == ["保固三年"]


def test_version_changes_on_every_write(store, embedder):
    assert store.version() == 0
    _add(store, embedder, "me", "a.md", [("", "一"), ("", "二")])
    _add(store, embedder, "me", "a.md", [("", "一"), ("", "二")])  # 取代：段落數與 id 都可能相同
    assert store.version() == 2
    store.delete_document("me", "a.md")
    store.clear()
    assert store.version() == 4


def test_cache_rebuilds_after_same_size_replace(store, embedder):
    retriever = Retriever(store, embedder)
    _add(store, embedder, "me", "a.md", [("", "保固兩年")])
    retriever.retrieve("保固", ["me"], 5, 0.0)
    _add(store, embedder, "me", "a.md", [("", "保固三年")])
    assert retriever.retrieve("保固", ["me"], 5, 0.0).sources[0].text == "保固三年"

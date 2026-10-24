import pytest

from rag_chatbot.db import connect
from rag_chatbot.embedder import EmbedderError
from rag_chatbot.ingest import (
    IngestError, Status, collection_files, ingest_collections, ingest_file,
    list_collections, sha256_file,
)
from rag_chatbot.vector_store import VectorStore
from tests.builders import make_csv, make_pdf
from tests.fakes import FakeEmbedder


@pytest.fixture
def store(project):
    conn = connect(project.paths.db)
    yield VectorStore(conn, "fake-embed")
    conn.close()


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def folders(project):
    root = project.paths.collections
    return {"me": root / "me", "product": root / "product"}


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_sha256_file(tmp_path):
    path = _write(tmp_path / "a.txt", "abc")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_list_collections_and_files(project, folders):
    (project.paths.collections / ".hidden").mkdir()
    _write(folders["me"] / "b.md", "x")
    _write(folders["me"] / "a.md", "x")
    _write(folders["me"] / ".DS_Store", "x")
    _write(folders["me"] / "~$履歷.docx", "x")
    (folders["me"] / "sub").mkdir()
    assert list_collections(project) == ["me", "product"]
    assert [p.name for p in collection_files(project, "me")] == ["a.md", "b.md"]
    with pytest.raises(IngestError, match="找不到文件集資料夾"):
        collection_files(project, "nope")


def test_list_collections_without_root(write_file, tmp_path):
    from rag_chatbot.config import load_config

    config = load_config(write_file("config.toml", ""))
    assert list_collections(config) == []


def test_add_unchanged_update(project, store, embedder, folders):
    path = _write(folders["me"] / "作品集.md", "## 小算盤\n記帳 App。")
    first = ingest_file(store, embedder, project, "me", path)
    assert (first.status, first.chunk_count) == (Status.ADDED, 1)

    calls = len(embedder.calls)
    again = ingest_file(store, embedder, project, "me", path)
    assert (again.status, again.chunk_count) == (Status.UNCHANGED, 1)
    assert len(embedder.calls) == calls  # 未變更時不呼叫 Embedder

    _write(path, "## 小算盤\n記帳 App。\n## 部落格\n技術文章。")
    updated = ingest_file(store, embedder, project, "me", path)
    assert (updated.status, updated.chunk_count) == (Status.UPDATED, 2)
    assert store.chunk_count() == 2


def test_embedding_text_has_prefix_and_db_has_raw_text(project, store, embedder, folders):
    path = _write(folders["me"] / "作品集.md", "## 小算盤\n記帳 App。")
    ingest_file(store, embedder, project, "me", path)
    assert embedder.calls[-1] == ["作品集 > 小算盤\n記帳 App。"]
    assert store.conn.execute("SELECT text, heading FROM chunks").fetchone()[:] == ("記帳 App。", "小算盤")


def test_long_text_is_chunked_but_faq_is_not(project, store, embedder, folders):
    long_text = "這是一句測試用的句子。" * 200  # 2,200 字
    md = ingest_file(store, embedder, project, "me", _write(folders["me"] / "long.md", long_text))
    assert md.chunk_count > 3
    csv_path = make_csv(folders["product"] / "faq.csv", [["question", "answer"], ["長答案？", long_text]])
    faq = ingest_file(store, embedder, project, "product", csv_path)
    assert faq.chunk_count == 1


def test_pdf_keeps_page(project, store, embedder, folders):
    path = make_pdf(folders["product"] / "手冊.pdf", ["簡介", "方案價格"])
    ingest_file(store, embedder, project, "product", path)
    pages = [r[0] for r in store.conn.execute("SELECT page FROM chunks ORDER BY chunk_index")]
    assert pages == [1, 2]


def test_failed_and_skipped_files(project, store, embedder, folders):
    bad = folders["me"] / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    result = ingest_file(store, embedder, project, "me", bad)
    assert result.status is Status.FAILED
    assert "無法開啟 PDF" in result.message

    txt = _write(folders["me"] / "notes.txt", "x")
    assert ingest_file(store, embedder, project, "me", txt).status is Status.SKIPPED
    assert store.list_documents() == []


def test_failed_update_keeps_old_version(project, store, embedder, folders):
    path = _write(folders["me"] / "a.md", "舊內容")
    ingest_file(store, embedder, project, "me", path)
    path.write_bytes(b"")  # 改成空檔，解析失敗
    assert ingest_file(store, embedder, project, "me", path).status is Status.FAILED
    assert store.conn.execute("SELECT text FROM chunks").fetchone()[0] == "舊內容"


def test_embedder_failure_keeps_old_version_and_raises(project, store, folders):
    path = _write(folders["me"] / "a.md", "舊內容")
    embedder = FakeEmbedder(fail_on_call=2)
    ingest_file(store, embedder, project, "me", path)
    _write(path, "新內容")
    with pytest.raises(EmbedderError):
        ingest_file(store, embedder, project, "me", path)
    assert store.get_document("me", "a.md").chunk_count == 1
    assert store.conn.execute("SELECT text FROM chunks").fetchone()[0] == "舊內容"


def test_ingest_collections(project, store, embedder, folders):
    _write(folders["me"] / "a.md", "內容一")
    (folders["me"] / "bad.docx").write_bytes(b"x")
    _write(folders["product"] / "b.md", "內容二")
    started, reported, progress = [], [], []
    results = ingest_collections(
        store, embedder, project,
        on_start=lambda c, f: started.append(f"{c}/{f}"),
        on_result=reported.append,
        progress=lambda d, t: progress.append((d, t)),
    )
    assert started == ["me/a.md", "me/bad.docx", "product/b.md"]
    assert reported == results
    assert [r.status for r in results] == [Status.ADDED, Status.FAILED, Status.ADDED]
    assert progress == [(1, 1), (1, 1)]

    only_product = ingest_collections(store, embedder, project, ["product"])
    assert [(r.filename, r.status) for r in only_product] == [("b.md", Status.UNCHANGED)]


def test_ingest_collections_model_checks(project, store, embedder, folders):
    with pytest.raises(IngestError, match="Embedder 模型"):
        ingest_collections(store, FakeEmbedder(model="other"), project)

    _write(folders["me"] / "a.md", "內容")
    ingest_collections(store, embedder, project)
    new_store = VectorStore(store.conn, "new-model")
    with pytest.raises(IngestError, match="ingest --rebuild"):
        ingest_collections(new_store, FakeEmbedder(model="new-model"), project)

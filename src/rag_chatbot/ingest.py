"""匯入流程：讀檔 → hash 比對 → 解析 → 切段 → 向量化 → 寫入（設計見 docs/02-design.md §4.1）。"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from rag_chatbot import loaders
from rag_chatbot.chunker import chunk_sections
from rag_chatbot.config import Config
from rag_chatbot.embedder import Embedder, ProgressCallback, embedding_text
from rag_chatbot.vector_store import VectorStore


class IngestError(Exception):
    """無法進行匯入（文件集不存在、embedding 模型不符等）。"""


class Status(Enum):
    ADDED = "新增"
    UPDATED = "已取代"
    UNCHANGED = "已是最新"
    FAILED = "失敗"
    SKIPPED = "略過"


@dataclass(frozen=True)
class FileResult:
    collection: str
    filename: str
    status: Status
    chunk_count: int = 0
    message: str = ""


# on_start(collection, filename)：開始處理某個檔案時呼叫，方便顯示進度
StartCallback = Callable[[str, str], None]
ResultCallback = Callable[[FileResult], None]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def list_collections(config: Config) -> list[str]:
    """data/collections/ 下的子資料夾名稱（排序）。"""
    root = config.paths.collections
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))


def collection_files(config: Config, collection: str) -> list[Path]:
    """文件集資料夾第一層的檔案（排序），略過隱藏檔與 Office 暫存檔（~$ 開頭）。"""
    folder = config.paths.collections / collection
    if not folder.is_dir():
        raise IngestError(f"找不到文件集資料夾：{folder}")
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and not p.name.startswith((".", "~$"))
    )


def ingest_file(
    store: VectorStore,
    embedder: Embedder,
    config: Config,
    collection: str,
    path: Path,
    progress: ProgressCallback | None = None,
) -> FileResult:
    """匯入單一檔案。文件本身的問題回傳 FAILED；Ollama、資料庫的問題直接丟出例外。"""
    filename = path.name
    file_type = loaders.file_type(path)
    if file_type is None:
        return FileResult(collection, filename, Status.SKIPPED, message="不支援的格式")

    sha = sha256_file(path)
    existing = store.get_document(collection, filename)
    if existing is not None and existing.sha256 == sha:
        return FileResult(collection, filename, Status.UNCHANGED, existing.chunk_count)

    try:
        sections = loaders.load(path)
    except loaders.LoaderError as e:
        return FileResult(collection, filename, Status.FAILED, message=str(e))

    chunks = chunk_sections(
        sections,
        config.chunking.chunk_size,
        config.chunking.chunk_overlap,
        split=file_type != "faq",
    )
    vectors = embedder.embed([embedding_text(filename, c.heading, c.text) for c in chunks], progress)
    info = store.replace_document(collection, filename, file_type, sha, chunks, vectors)
    status = Status.ADDED if existing is None else Status.UPDATED
    return FileResult(collection, filename, status, info.chunk_count)


def ingest_collections(
    store: VectorStore,
    embedder: Embedder,
    config: Config,
    collections: list[str] | None = None,
    on_start: StartCallback | None = None,
    on_result: ResultCallback | None = None,
    progress: ProgressCallback | None = None,
) -> list[FileResult]:
    """匯入指定文件集（預設全部）。每個檔案獨立處理，一個檔案失敗不影響其他檔案。"""
    if embedder.model != store.embed_model:
        raise IngestError(f"Embedder 模型 {embedder.model} 與設定 {store.embed_model} 不同")
    stored = store.embed_model_mismatch()
    if stored is not None:
        raise IngestError(
            f"資料庫的向量由 {stored} 產生，目前設定為 {store.embed_model}，請使用 ingest --rebuild 重新匯入全部文件"
        )

    names = list_collections(config) if collections is None else collections
    results = []
    for collection in names:
        for path in collection_files(config, collection):
            if on_start:
                on_start(collection, path.name)
            result = ingest_file(store, embedder, config, collection, path, progress)
            results.append(result)
            if on_result:
                on_result(result)
    return results

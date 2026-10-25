"""文件、段落與向量的存取，以及相似度搜尋（設計見 docs/02-design.md §4.1、§5）。"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime

import numpy as np

from rag_chatbot.db import get_meta, set_meta
from rag_chatbot.models import Chunk, Source

_EMBED_MODEL_KEY = "embed_model"
_VERSION_KEY = "chunks_version"


class VectorStoreError(Exception):
    """向量資料不一致（模型不符、維度不符等）。"""


@dataclass(frozen=True)
class DocumentInfo:
    id: int
    collection: str
    filename: str
    file_type: str
    sha256: str
    chunk_count: int
    imported_at: str


class VectorStore:
    def __init__(self, conn: sqlite3.Connection, embed_model: str) -> None:
        self.conn = conn
        self.embed_model = embed_model

    # ---------- embedding 模型 ----------

    def stored_embed_model(self) -> str | None:
        """資料庫中向量所用的模型；沒有任何段落時回傳 None。"""
        if self.chunk_count() == 0:
            return None
        return get_meta(self.conn, _EMBED_MODEL_KEY)

    def embed_model_mismatch(self) -> str | None:
        """資料庫向量的模型與目前設定不同時，回傳資料庫中的模型名稱。"""
        stored = self.stored_embed_model()
        return stored if stored is not None and stored != self.embed_model else None

    def _ensure_model(self) -> None:
        stored = self.embed_model_mismatch()
        if stored is not None:
            raise VectorStoreError(
                f"資料庫的向量由 {stored} 產生，目前設定為 {self.embed_model}，需要重新匯入全部文件"
            )

    # ---------- 文件 ----------

    def get_document(self, collection: str, filename: str) -> DocumentInfo | None:
        row = self.conn.execute(
            "SELECT * FROM documents WHERE collection = ? AND filename = ?", (collection, filename)
        ).fetchone()
        return _document(row) if row else None

    def list_documents(self, collection: str | None = None) -> list[DocumentInfo]:
        if collection is None:
            rows = self.conn.execute("SELECT * FROM documents ORDER BY collection, filename")
        else:
            rows = self.conn.execute(
                "SELECT * FROM documents WHERE collection = ? ORDER BY filename", (collection,)
            )
        return [_document(r) for r in rows]

    def chunk_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def replace_document(
        self,
        collection: str,
        filename: str,
        file_type: str,
        sha256: str,
        chunks: Sequence[Chunk],
        vectors: np.ndarray,
    ) -> DocumentInfo:
        """以單一交易新增或取代文件與其段落；任何錯誤都不會改動既有資料（D-019）。"""
        if len(chunks) == 0:
            raise VectorStoreError(f"{collection}/{filename}：沒有段落可寫入")
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise VectorStoreError(f"{collection}/{filename}：段落數與向量數不符")
        self._ensure_model()
        self._ensure_dim(vectors.shape[1], collection, filename)
        normalized = _normalize(vectors, f"{collection}/{filename}")

        imported_at = datetime.now().isoformat(timespec="seconds")
        with self.conn:
            self.conn.execute(
                "DELETE FROM documents WHERE collection = ? AND filename = ?", (collection, filename)
            )
            cursor = self.conn.execute(
                "INSERT INTO documents(collection, filename, file_type, sha256, chunk_count, imported_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (collection, filename, file_type, sha256, len(chunks), imported_at),
            )
            document_id = cursor.lastrowid
            self.conn.executemany(
                "INSERT INTO chunks(document_id, chunk_index, text, page, heading, embedding)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (document_id, c.chunk_index, c.text, c.page, c.heading, v.tobytes())
                    for c, v in zip(chunks, normalized)
                ],
            )
            set_meta(self.conn, _EMBED_MODEL_KEY, self.embed_model)
            self._bump_version()
        return DocumentInfo(document_id, collection, filename, file_type, sha256, len(chunks), imported_at)

    def delete_document(self, collection: str, filename: str) -> bool:
        """刪除文件與其段落；文件不存在時回傳 False。"""
        with self.conn:
            cursor = self.conn.execute(
                "DELETE FROM documents WHERE collection = ? AND filename = ?", (collection, filename)
            )
            self._bump_version()
        return cursor.rowcount > 0

    def clear(self) -> int:
        """刪除全部文件與段落（換 embedding 模型後重建用），回傳刪除的文件數。"""
        with self.conn:
            cursor = self.conn.execute("DELETE FROM documents")
            self._bump_version()
        return cursor.rowcount

    def _bump_version(self) -> None:
        # 段落異動的版本號；SQLite 會重複使用已刪除的 id，所以不能只看段落數與最大 id
        set_meta(self.conn, _VERSION_KEY, str(self.version() + 1))

    def _ensure_dim(self, dim: int, collection: str, filename: str) -> None:
        row = self.conn.execute(
            "SELECT length(c.embedding) AS size FROM chunks c JOIN documents d ON d.id = c.document_id"
            " WHERE NOT (d.collection = ? AND d.filename = ?) LIMIT 1",
            (collection, filename),
        ).fetchone()
        if row is not None and row["size"] != dim * 4:
            raise VectorStoreError(
                f"{collection}/{filename}：向量維度 {dim} 與資料庫中的 {row['size'] // 4} 不符"
            )

    # ---------- 搜尋 ----------

    def version(self) -> int:
        """段落異動的版本號：每次新增、取代、刪除、清空都加 1，供快取判斷是否需要重建（跨程序亦可）。"""
        return int(get_meta(self.conn, _VERSION_KEY) or 0)

    def load_chunks(self, collections: Sequence[str]) -> tuple[list[Source], np.ndarray]:
        """讀出指定文件集的全部段落（score 為 0）與正規化向量矩陣，順序一致。"""
        self._ensure_model()
        if not collections:
            return [], np.zeros((0, 0), dtype=np.float32)
        placeholders = ",".join("?" * len(collections))
        rows = self.conn.execute(
            "SELECT c.id, d.collection, d.filename, c.page, c.heading, c.text, c.embedding"
            " FROM chunks c JOIN documents d ON d.id = c.document_id"
            f" WHERE d.collection IN ({placeholders}) ORDER BY c.id",
            tuple(collections),
        ).fetchall()
        if not rows:
            return [], np.zeros((0, 0), dtype=np.float32)
        chunks = [
            Source(r["id"], r["collection"], r["filename"], r["page"], r["heading"], r["text"], 0.0)
            for r in rows
        ]
        matrix = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        return chunks, matrix

    def search(self, query: np.ndarray, collections: Sequence[str], top_k: int) -> list[Source]:
        """在指定文件集中以餘弦相似度搜尋，依分數由高到低回傳最多 top_k 筆（不做門檻過濾）。"""
        if top_k <= 0:
            raise ValueError("top_k 必須大於 0")
        chunks, matrix = self.load_chunks(collections)
        if not chunks:
            return []
        scores = cosine_scores(matrix, query)
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [replace(chunks[i], score=float(scores[i])) for i in order]


def cosine_scores(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    """matrix 的每列已正規化；回傳每列與 query 的餘弦相似度。"""
    q = np.asarray(query, dtype=np.float32).reshape(-1)
    if q.shape[0] != matrix.shape[1]:
        raise VectorStoreError(f"查詢向量維度 {q.shape[0]} 與資料庫中的 {matrix.shape[1]} 不符")
    return matrix @ _normalize(q[np.newaxis, :], "查詢")[0]


def _normalize(vectors: np.ndarray, label: str) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0) or not np.all(np.isfinite(norms)):
        raise VectorStoreError(f"{label}：向量為零或含無效值")
    return (vectors / norms).astype("<f4")


def _document(row: sqlite3.Row) -> DocumentInfo:
    return DocumentInfo(
        id=row["id"],
        collection=row["collection"],
        filename=row["filename"],
        file_type=row["file_type"],
        sha256=row["sha256"],
        chunk_count=row["chunk_count"],
        imported_at=row["imported_at"],
    )

"""命令列工具（設計見 docs/02-design.md §4.4）。

python -X utf8 -m rag_chatbot [--config 設定檔] <指令>
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Callable, Sequence

from rag_chatbot.config import DEFAULT_CONFIG_PATH, Config, ConfigError, load_config
from rag_chatbot.db import DatabaseError, connect
from rag_chatbot.embedder import Embedder, EmbedderError, OllamaEmbedder
from rag_chatbot.ingest import FileResult, IngestError, Status, ingest_collections, list_collections
from rag_chatbot.profiles import ProfileError, list_profiles, load_profile
from rag_chatbot.vector_store import VectorStore, VectorStoreError

EmbedderFactory = Callable[[Config], Embedder]

_PREVIEW_CHARS = 80
_EXPECTED_ERRORS = (ConfigError, ProfileError, DatabaseError, IngestError, EmbedderError, VectorStoreError)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag_chatbot", description="RAG 問答系統命令列工具")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="設定檔路徑（預設 config.toml）")
    sub = parser.add_subparsers(dest="command", required=True, metavar="指令")

    sub.add_parser("profiles", help="列出情境與其文件集")

    p = sub.add_parser("ingest", help="匯入全部或指定文件集")
    p.add_argument("collection", nargs="?", help="文件集名稱（省略時匯入全部）")
    p.add_argument("--rebuild", action="store_true", help="先清空資料庫中全部文件再匯入")

    p = sub.add_parser("list", help="列出已匯入的文件")
    p.add_argument("collection", nargs="?", help="文件集名稱（省略時列出全部）")

    p = sub.add_parser("search", help="顯示最相關的段落與分數")
    p.add_argument("question", help="問題")
    p.add_argument("--profile", help="情境名稱（預設為設定檔中的 default_profile）")
    p.add_argument("--top-k", type=int, help="顯示筆數（預設依情境設定）")

    p = sub.add_parser("delete", help="從資料庫刪除文件（不刪原始檔）")
    p.add_argument("collection", help="文件集名稱")
    p.add_argument("filename", help="檔名")
    return parser


def main(argv: Sequence[str] | None = None, embedder_factory: EmbedderFactory | None = None) -> int:
    # 主控台不是 UTF-8 時，無法顯示的字元以 ? 代替，避免整個指令失敗
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    args = build_parser().parse_args(argv)
    factory = embedder_factory or OllamaEmbedder.from_config
    try:
        config = load_config(args.config)
        if args.command == "profiles":
            return _cmd_profiles(config)
        conn = connect(config.paths.db)
        try:
            store = VectorStore(conn, config.ollama.embed_model)
            if args.command == "ingest":
                return _cmd_ingest(config, store, factory(config), args.collection, args.rebuild)
            if args.command == "list":
                return _cmd_list(config, store, args.collection)
            if args.command == "search":
                return _cmd_search(config, store, factory(config), args.question, args.profile, args.top_k)
            return _cmd_delete(store, args.collection, args.filename)
        finally:
            conn.close()
    except _EXPECTED_ERRORS as e:
        print(f"錯誤：{e}", file=sys.stderr)
        return 1


def _cmd_profiles(config: Config) -> int:
    profiles = list_profiles(config)
    if not profiles:
        print(f"沒有情境設定：{config.paths.profiles}")
        return 0
    for p in profiles:
        default = "（預設）" if p.name == config.default_profile else ""
        print(f"{p.name}{default}：{p.title}，介紹對象 {p.subject}")
        missing = set(p.missing_collections(config))
        for c in p.collections:
            print(f"  - {c}{'（資料夾不存在）' if c in missing else ''}")
    return 0


def _cmd_ingest(config: Config, store: VectorStore, embedder: Embedder, collection: str | None, rebuild: bool) -> int:
    if rebuild:
        print(f"已清空資料庫中的 {store.clear()} 份文件")
    collections = [collection] if collection else None
    if collections is None and not list_collections(config):
        print(f"沒有文件集：請在 {config.paths.collections} 下建立子資料夾並放入文件")
        return 0

    interactive = sys.stdout.isatty()

    def on_start(c: str, filename: str) -> None:
        if interactive:
            print(f"  處理中：{c}/{filename}", end="\r", flush=True)

    def progress(done: int, total: int) -> None:
        if interactive:
            print(f"  向量化：{done}/{total} 段", end="\r", flush=True)

    def on_result(r: FileResult) -> None:
        if interactive:
            print(" " * 60, end="\r")
        detail = f"（{r.chunk_count} 段）" if r.chunk_count else ""
        message = f"：{r.message}" if r.message else ""
        print(f"[{r.status.value}] {r.collection}/{r.filename}{detail}{message}", flush=True)

    results = ingest_collections(store, embedder, config, collections, on_start, on_result, progress)
    counts = Counter(r.status for r in results)
    print("完成：" + "、".join(f"{s.value} {counts[s]}" for s in Status))
    return 1 if counts[Status.FAILED] else 0


def _cmd_list(config: Config, store: VectorStore, collection: str | None) -> int:
    _warn_model_mismatch(store)
    documents = store.list_documents(collection)
    if not documents:
        print("沒有已匯入的文件")
        return 0
    for d in documents:
        missing = "" if (config.paths.collections / d.collection / d.filename).is_file() else "（原始檔不存在）"
        print(f"{d.collection}/{d.filename}  {d.file_type}  {d.chunk_count} 段  {d.imported_at}{missing}")
    print(f"共 {len(documents)} 份文件、{sum(d.chunk_count for d in documents)} 段")
    return 0


def _cmd_search(
    config: Config,
    store: VectorStore,
    embedder: Embedder,
    question: str,
    profile_name: str | None,
    top_k: int | None,
) -> int:
    if top_k is not None and top_k <= 0:
        raise ConfigError("--top-k 必須大於 0")
    profile = load_profile(config, profile_name)
    top_k = top_k or profile.resolved_top_k(config)
    min_score = profile.resolved_min_score(config)
    missing = profile.missing_collections(config)
    if missing:
        print(f"注意：文件集資料夾不存在：{', '.join(missing)}")

    query = embedder.embed([question])[0]
    results = store.search(query, profile.collections, top_k)
    print(f"情境 {profile.name}（文件集：{', '.join(profile.collections)}），門檻 {min_score:.2f}")
    if not results:
        print("沒有任何段落")
        return 0
    for rank, s in enumerate(results, start=1):
        location = f" 第 {s.page} 頁" if s.page else ""
        heading = f" [{s.heading}]" if s.heading else ""
        below = "  ← 低於門檻" if s.score < min_score else ""
        preview = " ".join(s.text.split())[:_PREVIEW_CHARS]
        print(f"{rank}. {s.score:.4f}  {s.collection}/{s.filename}{location}{heading}{below}")
        print(f"   {preview}")
    return 0


def _cmd_delete(store: VectorStore, collection: str, filename: str) -> int:
    if store.delete_document(collection, filename):
        print(f"已刪除：{collection}/{filename}（原始檔保留）")
        return 0
    print(f"錯誤：資料庫中沒有 {collection}/{filename}", file=sys.stderr)
    return 1


def _warn_model_mismatch(store: VectorStore) -> None:
    stored = store.embed_model_mismatch()
    if stored is not None:
        print(f"注意：資料庫的向量由 {stored} 產生，目前設定為 {store.embed_model}，請執行 ingest --rebuild")

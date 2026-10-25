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
from rag_chatbot.history import HistoryError
from rag_chatbot.ingest import FileResult, IngestError, Status, ingest_collections, list_collections
from rag_chatbot.llm import LLM, LLMError, OllamaLLM
from rag_chatbot.models import ANSWERED, GREETING, REFUSED, Answer
from rag_chatbot.profiles import ProfileError, list_profiles, load_profile
from rag_chatbot.prompts import PromptError
from rag_chatbot.rag import RagService, source_label
from rag_chatbot.retriever import Retriever
from rag_chatbot.vector_store import VectorStore, VectorStoreError

EmbedderFactory = Callable[[Config], Embedder]
LLMFactory = Callable[[Config], LLM]

_PREVIEW_CHARS = 80
_EXPECTED_ERRORS = (
    ConfigError, ProfileError, DatabaseError, IngestError, EmbedderError, VectorStoreError,
    LLMError, HistoryError, PromptError,
)
_KIND_LABELS = {ANSWERED: "回答", REFUSED: "拒答", GREETING: "問候"}
_QUIT_WORDS = {"exit", "quit", "q", "離開", "結束"}


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

    p = sub.add_parser("ask", help="回答一個問題，列出依據")
    p.add_argument("question", help="問題")
    p.add_argument("--profile", help="情境名稱（預設為設定檔中的 default_profile）")
    p.add_argument("--conversation", type=int, help="接續指定的對話編號（多輪追問）")
    p.add_argument("--prompt-version", type=int, help="使用指定的 Prompt 版本（預設為啟用中的版本）")

    p = sub.add_parser("chat", help="互動式對話（連續追問，輸入 exit 離開）")
    p.add_argument("--profile", help="情境名稱（預設為設定檔中的 default_profile）")
    p.add_argument("--sources", action="store_true", help="每則回答後列出依據")
    return parser


def main(
    argv: Sequence[str] | None = None,
    embedder_factory: EmbedderFactory | None = None,
    llm_factory: LLMFactory | None = None,
) -> int:
    # 主控台不是 UTF-8 時，無法顯示的字元以 ? 代替，避免整個指令失敗
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    args = build_parser().parse_args(argv)
    factory = embedder_factory or OllamaEmbedder.from_config
    make_llm = llm_factory or OllamaLLM.from_config
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
            if args.command == "delete":
                return _cmd_delete(store, args.collection, args.filename)
            service = RagService(config, store, factory(config), make_llm(config))
            if args.command == "ask":
                return _cmd_ask(service, args.question, args.profile, args.conversation, args.prompt_version)
            return _cmd_chat(service, args.profile, args.sources)
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

    # 門檻設為 0 以列出全部結果，另外標示是否達到真正的門檻；不套用參考資料長度上限
    result = Retriever(store, embedder).retrieve(question, profile.collections, top_k, min_score=0.0)
    print(f"情境 {profile.name}（文件集：{', '.join(profile.collections)}），門檻 {min_score:.2f}")
    if result.top_score is None:
        print("沒有任何段落")
        return 0
    verdict = "送給模型回答" if result.top_score >= min_score else "低於門檻，會直接拒答"
    print(f"語意最高分 {result.top_score:.4f} → {verdict}")
    for rank, s in enumerate(result.sources, start=1):
        location = f" 第 {s.page} 頁" if s.page else ""
        heading = f" [{s.heading}]" if s.heading else ""
        preview = " ".join(s.text.split())[:_PREVIEW_CHARS]
        print(f"{rank}. {s.score:.4f}  {s.collection}/{s.filename}{location}{heading}")
        print(f"   {preview}")
    return 0


def _cmd_delete(store: VectorStore, collection: str, filename: str) -> int:
    if store.delete_document(collection, filename):
        print(f"已刪除：{collection}/{filename}（原始檔保留）")
        return 0
    print(f"錯誤：資料庫中沒有 {collection}/{filename}", file=sys.stderr)
    return 1


def _cmd_ask(
    service: RagService,
    question: str,
    profile: str | None,
    conversation_id: int | None,
    prompt_version: int | None,
) -> int:
    if not question.strip():
        print("錯誤：問題不可為空白", file=sys.stderr)
        return 1
    answer = service.answer(question, profile, conversation_id, prompt_version)
    _print_answer(answer, with_sources=True)
    print(f"（對話 #{answer.conversation_id}；接續追問請加 --conversation {answer.conversation_id}）")
    return 0


def _cmd_chat(service: RagService, profile: str | None, with_sources: bool) -> int:
    p = load_profile(service.config, profile)
    print(f"情境 {p.name}：{p.title}。輸入問題後按 Enter；輸入 exit 或直接按 Enter 離開。")
    conversation_id = None
    while True:
        try:
            question = input("\n您：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question.lower() in _QUIT_WORDS:
            break
        print("（思考中…）", flush=True)
        try:
            answer = service.answer(question, p.name, conversation_id)
        except (EmbedderError, LLMError) as e:
            print(f"錯誤：{e}", file=sys.stderr)
            continue
        conversation_id = answer.conversation_id
        _print_answer(answer, with_sources)
    if conversation_id is not None:
        print(f"對話已保存（#{conversation_id}）")
    return 0


def _print_answer(answer: Answer, with_sources: bool) -> None:
    info = [_KIND_LABELS.get(answer.kind, answer.kind), f"{answer.latency_ms / 1000:.1f} 秒"]
    if answer.prompt_version is not None:
        info.append(f"Prompt v{answer.prompt_version}")
    if answer.top_score is not None:
        info.append(f"語意最高分 {answer.top_score:.3f}")
    print(f"[{'｜'.join(info)}]")
    print(answer.text)
    if with_sources and answer.sources:
        print("\n回答依據：")
        for i, s in enumerate(answer.sources, start=1):
            print(f"  [{i}] {s.score:.3f}  {s.collection}/{source_label(s)}")


def _warn_model_mismatch(store: VectorStore) -> None:
    stored = store.embed_model_mismatch()
    if stored is not None:
        print(f"注意：資料庫的向量由 {stored} 產生，目前設定為 {store.embed_model}，請執行 ingest --rebuild")

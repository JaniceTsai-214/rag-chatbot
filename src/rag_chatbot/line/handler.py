"""LINE 事件處理（docs/05-line.md §4.1、§4.2）。

accept() 在 webhook 執行緒呼叫：去重、過濾、頻率限制，需要等待的工作排入佇列。
run_worker() 在背景執行緒：一次處理一題，呼叫 answer() 後以 reply 回覆，逾時或失敗改用 push。
"""

from __future__ import annotations

import queue
import re
import threading
import time
from collections import OrderedDict, defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from rag_chatbot.config import Config
from rag_chatbot.embedder import EmbedderError
from rag_chatbot.history import HistoryError
from rag_chatbot.line.api import LineApi, LineApiError
from rag_chatbot.line.users import LineUsers
from rag_chatbot.llm import LLMError
from rag_chatbot.profiles import load_profile
from rag_chatbot.rag import RagService
from rag_chatbot.smalltalk import classify
from rag_chatbot.vector_store import VectorStoreError

NON_TEXT_TEXT = "目前只能回答文字問題，歡迎直接輸入您想了解的內容。"
RATE_LIMIT_TEXT = "您的提問有點頻繁，請稍候幾分鐘再問。"
BUSY_TEXT = "目前詢問的人較多，請稍候再問。"
ERROR_TEXT = "系統暫時無法回答，請稍後再試，或來信 {contact} 詢問。"
SEEN_EVENTS = 1000  # 去重時記住的事件數

FOLLOW = "follow"
TEXT = "text"


@dataclass(frozen=True)
class Task:
    kind: str  # FOLLOW 或 TEXT
    user_id: str
    reply_token: str
    received_at: float  # clock() 的時間
    text: str = ""


def format_for_line(text: str) -> str:
    """移除 [1]、[1][2]、[1]、[3] 等引用標記（LINE 不顯示依據，L-07），並整理多餘空白。"""
    text = re.sub(r"\[\d+\](?:\s*[、,，]?\s*\[\d+\])*", "", text)
    text = re.sub(r"[ \t]+([。，、！？；：）」])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


class LineBot:
    def __init__(
        self,
        config: Config,
        api: LineApi,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = datetime.now,
        log: Callable[[str], None] = print,
    ) -> None:
        self.config = config
        self.api = api
        self.queue: queue.Queue[Task] = queue.Queue()
        self._clock = clock
        self._now = now
        self._log = log
        self._lock = threading.Lock()
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._recent: defaultdict[str, deque[float]] = defaultdict(deque)

    # ---------- webhook 執行緒 ----------

    def accept(self, payload: dict[str, Any]) -> None:
        for event in payload.get("events", []):
            with self._lock:
                result = self._triage(event)
            if isinstance(result, Task):
                self.queue.put(result)
            elif result is not None:
                token, text = result
                self._safe_reply(token, text)

    def _triage(self, event: dict[str, Any]) -> Task | tuple[str, str] | None:
        """回傳要排入佇列的 Task、要立即回覆的 (reply token, 文字)，或 None（忽略）。"""
        event_id = event.get("webhookEventId")
        if event_id:
            if event_id in self._seen:
                return None
            self._seen[event_id] = None
            if len(self._seen) > SEEN_EVENTS:
                self._seen.popitem(last=False)

        source = event.get("source", {})
        user_id, token = source.get("userId"), event.get("replyToken")
        if source.get("type") != "user" or not user_id or not token:
            return None  # 群組、多人聊天，或不能回覆的事件

        if event.get("type") == "follow":
            return Task(FOLLOW, user_id, token, self._clock())
        if event.get("type") != "message":
            return None
        message = event.get("message", {})
        if message.get("type") != "text":
            return token, NON_TEXT_TEXT
        if self._rate_limited(user_id):
            return token, RATE_LIMIT_TEXT
        if self.queue.qsize() >= self.config.line.max_queue:
            return token, BUSY_TEXT
        return Task(TEXT, user_id, token, self._clock(), message.get("text", ""))

    def _rate_limited(self, user_id: str) -> bool:
        now = self._clock()
        window = self.config.line.rate_limit_minutes * 60
        recent = self._recent[user_id]
        while recent and now - recent[0] >= window:
            recent.popleft()
        if len(recent) >= self.config.line.rate_limit_count:
            return True
        recent.append(now)
        return False

    # ---------- 背景執行緒 ----------

    def run_worker(self, service_factory: Callable[[], RagService], stop: threading.Event) -> None:
        """SQLite 連線不能跨執行緒，RagService 在本執行緒內建立。"""
        service = service_factory()
        users = LineUsers(service.history.conn)
        try:
            while not stop.is_set():
                try:
                    task = self.queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                try:
                    self.process(task, service, users)
                except Exception as e:  # 任何意外都不能讓背景執行緒停止
                    self._log(f"處理失敗：{e!r}")
                finally:
                    self.queue.task_done()
        finally:
            service.history.conn.close()

    def process(self, task: Task, service: RagService, users: LineUsers) -> None:
        profile = load_profile(self.config, self.config.line.profile)
        if task.kind == FOLLOW:
            self._deliver(task, profile.greeting_text)
            self._log(f"{self._stamp(task)} 加好友")
            return

        if classify(task.text) is None:
            try:
                self.api.start_loading(task.user_id)
            except LineApiError as e:
                self._log(f"無法顯示輸入中動畫：{e}")

        idle = self.config.line.conversation_idle_minutes
        conversation_id = users.active_conversation(task.user_id, self._now(), idle)
        try:
            try:
                answer = service.answer(task.text, profile.name, conversation_id)
            except HistoryError:  # 對話已被刪除：開新對話
                answer = service.answer(task.text, profile.name, None)
        except (LLMError, EmbedderError, VectorStoreError) as e:
            self._log(f"{self._stamp(task)} 回答失敗：{e}")
            self._deliver(task, ERROR_TEXT.format(contact=profile.contact))
            return
        users.touch(task.user_id, answer.conversation_id, self._now())
        via = self._deliver(task, format_for_line(answer.text))
        self._log(
            f"{self._stamp(task)} {answer.kind} {answer.latency_ms / 1000:.1f} 秒 {via}｜{task.text[:30]}"
        )

    def _deliver(self, task: Task, text: str) -> str:
        """期限內用 reply（免費）；逾時或 reply 失敗改用 push（計入每月額度）。回傳實際方式。"""
        if self._clock() - task.received_at < self.config.line.reply_deadline_seconds:
            try:
                self.api.reply(task.reply_token, [text])
                return "reply"
            except LineApiError as e:
                self._log(f"reply 失敗，改用 push：{e}")
        try:
            self.api.push(task.user_id, [text])
            return "push"
        except LineApiError as e:
            self._log(f"push 失敗：{e}")
            return "失敗"

    def _safe_reply(self, token: str, text: str) -> None:
        try:
            self.api.reply(token, [text])
        except LineApiError as e:
            self._log(f"reply 失敗：{e}")

    def _stamp(self, task: Task) -> str:
        return f"{self._now():%H:%M:%S} 訪客…{task.user_id[-4:]}"

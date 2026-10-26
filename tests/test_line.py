import itertools
import json
from datetime import datetime, timedelta

import httpx
import pytest
from fastapi.testclient import TestClient

from rag_chatbot.config import ConfigError, load_config
from rag_chatbot.db import connect
from rag_chatbot.env import load_env
from rag_chatbot.line.api import MAX_MESSAGES, MAX_TEXT_CHARS, HttpLineApi, LineApiError
from rag_chatbot.line.app import create_app
from rag_chatbot.line.handler import (
    BUSY_TEXT, ERROR_TEXT, FOLLOW, NON_TEXT_TEXT, RATE_LIMIT_TEXT, TEXT, LineBot, Task, format_for_line,
)
from rag_chatbot.line.runner import LineSetupError, find_cloudflared, read_secrets, wait_for_tunnel_url
from rag_chatbot.line.signature import compute_signature, verify_signature
from rag_chatbot.line.users import LineUsers
from rag_chatbot.models import Chunk
from rag_chatbot.rag import RagService
from rag_chatbot.vector_store import VectorStore
from tests.fakes import FakeEmbedder, FakeLLM

SECRET = "test-secret"
T0 = datetime(2026, 10, 25, 12, 0, 0)


class FakeLineApi:
    def __init__(self, fail_reply=False, fail_push=False, fail_loading=False):
        self.calls = []
        self.fail_reply, self.fail_push, self.fail_loading = fail_reply, fail_push, fail_loading

    def reply(self, token, texts):
        self.calls.append(("reply", token, list(texts)))
        if self.fail_reply:
            raise LineApiError(400, "Invalid reply token")

    def push(self, user_id, texts):
        self.calls.append(("push", user_id, list(texts)))
        if self.fail_push:
            raise LineApiError(429, "quota")

    def start_loading(self, user_id, seconds=60):
        self.calls.append(("loading", user_id, seconds))
        if self.fail_loading:
            raise LineApiError(400, "not one-on-one")


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _event(kind="message", text="進階版價格", user="U1234567890", token="tok", event_id=None, source="user", message_type="text"):
    event = {"type": kind, "replyToken": token, "source": {"type": source, "userId": user}}
    if event_id:
        event["webhookEventId"] = event_id
    if kind == "message":
        event["message"] = {"type": message_type, "text": text} if message_type == "text" else {"type": message_type}
    return event


@pytest.fixture
def env(project):
    conn = connect(project.paths.db)
    store = VectorStore(conn, project.ollama.embed_model)
    embedder = FakeEmbedder(model=project.ollama.embed_model)
    chunks = [Chunk(0, "進階版價格每月 60 元", page=2)]
    store.replace_document("product", "手冊.pdf", "pdf", "h", chunks, embedder.embed(["進階版價格每月 60 元"]))
    llm = FakeLLM("進階版每月 60 元 [1]。")
    ticks = itertools.count(0, 1.5)
    service = RagService(project, store, embedder, llm, clock=lambda: next(ticks))
    api, clock, logs = FakeLineApi(), Clock(), []
    now = {"value": T0}
    bot = LineBot(project, api, clock=clock, now=lambda: now["value"], log=logs.append)
    yield bot, service, LineUsers(conn), api, clock, now, logs, llm
    conn.close()


# ---------- 簽章、.env、設定 ----------

def test_signature():
    body = b'{"events":[]}'
    signature = compute_signature(body, SECRET)
    assert verify_signature(body, signature, SECRET)
    assert not verify_signature(body + b" ", signature, SECRET)
    assert not verify_signature(body, signature, "other")
    assert not verify_signature(body, None, SECRET)


def test_load_env(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text('# 註解\nA=1\nB = "two"\nC=\'3\'\nnot a pair\n\nD=x=y\n', encoding="utf-8")
    monkeypatch.delenv("A", raising=False)
    monkeypatch.setenv("B", "from-env")
    assert load_env(path) == {"A": "1", "B": "from-env", "C": "3", "D": "x=y"}
    assert load_env(tmp_path / "missing") == {}


def test_line_config(write_file):
    config = load_config(write_file("c.toml", "[line]\nport = 9000\n"))
    assert config.line.port == 9000 and config.line.profile == "intro"
    for bad in ("port = 0", "max_queue = 0", "reply_deadline_seconds = -1"):
        with pytest.raises(ConfigError, match=r"\[line\]"):
            load_config(write_file("c.toml", f"[line]\n{bad}\n"))


# ---------- LINE API ----------

def _api(handler):
    return HttpLineApi("token-123", client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_api_requests():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path, request.headers["authorization"],
                     json.loads(request.content) if request.content else None))
        if request.method == "GET":
            return httpx.Response(200, json={"endpoint": "https://x/callback", "active": True})
        if request.url.path.endswith("/test"):
            return httpx.Response(200, json={"success": True})
        return httpx.Response(200, content=b"")

    api = _api(handler)
    api.reply("tok", ["一", "二"])
    api.push("U1", ["字" * (MAX_TEXT_CHARS + 10)] * (MAX_MESSAGES + 1))
    api.start_loading("U1")
    api.set_webhook("https://x/callback")
    assert api.webhook_info()["active"] is True
    assert api.test_webhook() == {"success": True}

    assert seen[0] == ("POST", "/v2/bot/message/reply", "Bearer token-123",
                       {"replyToken": "tok", "messages": [{"type": "text", "text": "一"}, {"type": "text", "text": "二"}]})
    push = seen[1][3]
    assert len(push["messages"]) == MAX_MESSAGES and len(push["messages"][0]["text"]) == MAX_TEXT_CHARS
    assert seen[2][1:] == ("/v2/bot/chat/loading/start", "Bearer token-123", {"chatId": "U1", "loadingSeconds": 60})
    assert seen[3][:2] == ("PUT", "/v2/bot/channel/webhook/endpoint")


def test_api_errors():
    api = _api(lambda request: httpx.Response(400, json={"message": "Invalid reply token"}))
    with pytest.raises(LineApiError, match="400：Invalid reply token") as error:
        api.reply("tok", ["x"])
    assert error.value.status == 400

    api = _api(lambda request: httpx.Response(500, text="oops"))
    with pytest.raises(LineApiError, match="500：oops"):
        api.push("U1", ["x"])

    def boom(request):
        raise httpx.ConnectError("down")

    with pytest.raises(LineApiError, match="無法連線到 LINE"):
        _api(boom).start_loading("U1")


# ---------- 訪客與對話 ----------

def test_line_users(env):
    _, service, users, *_ = env
    assert users.active_conversation("U1", T0, 30) is None
    c1 = service.history.create_conversation("intro", "a").id
    users.touch("U1", c1, T0)
    assert users.active_conversation("U1", T0 + timedelta(minutes=30), 30) == c1
    assert users.active_conversation("U1", T0 + timedelta(minutes=31), 30) is None
    c2 = service.history.create_conversation("intro", "b").id
    users.touch("U1", c2, T0 + timedelta(minutes=40))
    assert users.active_conversation("U1", T0 + timedelta(minutes=41), 30) == c2


# ---------- 回答整理 ----------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("每月 60 元。 [2]", "每月 60 元。"),
        ("包括：  [1] SAP、鼎新 [3]。", "包括： SAP、鼎新。"),
        ("共用。[1][2]", "共用。"),
        ("功能等。[1]、[3]、[4]", "功能等。"),
        ("- 重點一 [1]\n\n\n\n- 重點二 [2]", "- 重點一\n\n- 重點二"),
        ("沒有引用", "沒有引用"),
    ],
)
def test_format_for_line(text, expected):
    assert format_for_line(text) == expected


# ---------- accept（webhook 執行緒） ----------

def test_accept_queues_text_and_follow(env):
    bot, *_ = env
    bot.accept({"events": [_event(text="問題一"), _event("follow", token="t2")]})
    tasks = [bot.queue.get_nowait() for _ in range(2)]
    assert [(t.kind, t.text, t.reply_token) for t in tasks] == [(TEXT, "問題一", "tok"), (FOLLOW, "", "t2")]


def test_accept_ignores_groups_duplicates_and_other_events(env):
    bot, _, _, api, *_ = env
    bot.accept({"events": [
        _event(source="group"),
        _event("unfollow"),
        _event(token=None),
        _event(event_id="E1"),
        _event(event_id="E1"),
    ]})
    assert bot.queue.qsize() == 1 and api.calls == []
    bot.accept({})


def test_accept_dedupe_cache_is_bounded(env, monkeypatch):
    bot, *_ = env
    monkeypatch.setattr("rag_chatbot.line.handler.SEEN_EVENTS", 2)
    accepted = 0
    for event_id in ("E1", "E2", "E3", "E1"):  # E3 進來時 E1 被擠出快取，所以第二個 E1 不算重複
        bot.accept({"events": [_event(user=event_id, event_id=event_id)]})
        accepted += bot.queue.qsize()
        while not bot.queue.empty():
            bot.queue.get_nowait()
    assert accepted == 4


def test_accept_replies_immediately(env):
    bot, _, _, api, *_ = env
    bot.accept({"events": [_event(message_type="sticker", token="s")]})
    assert api.calls == [("reply", "s", [NON_TEXT_TEXT])]


def test_accept_rate_limit(env):
    bot, _, _, api, clock, *_ = env
    for _ in range(10):
        bot.accept({"events": [_event()]})
        bot.queue.get_nowait()
    bot.accept({"events": [_event(token="x")]})
    assert api.calls == [("reply", "x", [RATE_LIMIT_TEXT])]
    bot.accept({"events": [_event(user="U-other")]})  # 其他訪客不受影響
    assert bot.queue.qsize() == 1
    clock.t = 600  # 10 分鐘後恢復
    bot.accept({"events": [_event()]})
    assert bot.queue.qsize() == 2


def test_accept_busy(env):
    bot, _, _, api, *_ = env
    for i in range(3):
        bot.accept({"events": [_event(user=f"U{i}")]})
    bot.accept({"events": [_event(user="U9", token="b")]})
    assert bot.queue.qsize() == 3
    assert api.calls == [("reply", "b", [BUSY_TEXT])]


def test_accept_reply_failure_is_logged(env):
    bot, *_, logs, _ = env
    bot.api.fail_reply = True
    bot.accept({"events": [_event(message_type="image")]})
    assert any("reply 失敗" in line for line in logs)


# ---------- process（背景執行緒） ----------

def _task(text="進階版價格", received_at=0.0, user="U1234567890", kind=TEXT):
    return Task(kind, user, "tok", received_at, text)


def test_process_answer(env):
    bot, service, users, api, _, _, logs, llm = env
    bot.process(_task(), service, users)
    assert api.calls == [("loading", "U1234567890", 60), ("reply", "tok", ["進階版每月 60 元。"])]
    conversation_id = users.active_conversation("U1234567890", T0, 30)
    assert service.history.get_conversation(conversation_id).profile == "intro"
    assert "answered" in logs[-1] and "reply" in logs[-1] and "…7890" in logs[-1]


def test_process_follow_and_smalltalk(env):
    bot, service, users, api, *_ = env
    bot.process(_task(kind=FOLLOW, text=""), service, users)
    bot.process(_task(text="你好"), service, users)
    greeting = "您好，我是王小明的介紹助理，歡迎詢問相關問題。"
    assert api.calls == [("reply", "tok", [greeting]), ("reply", "tok", [greeting])]  # 問候不顯示動畫


def test_process_continues_conversation_until_idle(env):
    bot, service, users, _, _, now, _, llm = env
    bot.process(_task("進階版價格"), service, users)
    first = users.active_conversation("U1234567890", now["value"], 30)
    now["value"] = T0 + timedelta(minutes=10)
    bot.process(_task("那進階版呢"), service, users)
    assert users.active_conversation("U1234567890", now["value"], 30) == first
    assert "先前的問題：\n- 進階版價格" in llm.calls[-1][1]["content"]

    now["value"] = T0 + timedelta(minutes=50)
    bot.process(_task("進階版價格"), service, users)
    assert users.active_conversation("U1234567890", now["value"], 30) != first


def test_process_recovers_from_conversation_of_other_profile(env):
    bot, service, users, api, *_ = env
    other = service.history.create_conversation("me", "x").id  # 例如更改了 [line] profile
    users.touch("U1234567890", other, T0)
    bot.process(_task(), service, users)
    assert api.calls[-1][0] == "reply"
    assert users.active_conversation("U1234567890", T0, 30) not in (None, other)


def test_deleted_conversation_starts_new(env):
    _, service, users, *_ = env
    c = service.history.create_conversation("intro", "x").id
    users.touch("U1", c, T0)
    with service.history.conn:
        service.history.conn.execute("DELETE FROM conversations WHERE id = ?", (c,))
    assert users.active_conversation("U1", T0, 30) is None


def test_process_push_after_deadline_or_reply_failure(env):
    bot, service, users, api, clock, *_ = env
    clock.t = 60  # 已超過 50 秒
    bot.process(_task(received_at=0), service, users)
    assert api.calls[-1] == ("push", "U1234567890", ["進階版每月 60 元。"])

    api.calls.clear()
    clock.t = 0
    api.fail_reply = True
    bot.process(_task(), service, users)
    assert [c[0] for c in api.calls] == ["loading", "reply", "push"]


def test_process_all_delivery_failed(env):
    bot, service, users, api, _, _, logs, _ = env
    api.fail_reply = api.fail_push = api.fail_loading = True
    bot.process(_task(), service, users)
    assert any("無法顯示輸入中動畫" in line for line in logs)
    assert any("push 失敗" in line for line in logs)
    assert "失敗｜" in logs[-1]


def test_process_llm_error(env):
    bot, service, users, api, _, _, logs, llm = env
    llm.fail = True
    bot.process(_task(), service, users)
    assert api.calls[-1] == ("reply", "tok", [ERROR_TEXT.format(contact="test@example.com")])
    assert any("回答失敗" in line for line in logs)


def test_run_worker_processes_queue_and_survives_errors(env):
    import threading

    bot, service, _, api, *_ , logs, _ = env
    bot.queue.put(Task("unknown-kind-but-text", "U1", "tok", 0.0, "進階版價格"))
    bot.queue.put(_task())
    stop = threading.Event()

    calls = {"n": 0}
    original = bot.process

    def flaky(task, svc, users):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("意外錯誤")
        original(task, svc, users)

    def make_service():  # SQLite 連線必須在背景執行緒內建立
        config = service.config
        store = VectorStore(connect(config.paths.db), config.ollama.embed_model)
        return RagService(config, store, service.retriever.embedder, service.llm)

    bot.process = flaky
    thread = threading.Thread(target=bot.run_worker, args=(make_service, stop))
    thread.start()
    bot.queue.join()
    stop.set()
    thread.join(timeout=5)
    assert any("處理失敗" in line for line in logs)
    assert api.calls[-1][0] == "reply"


# ---------- webhook ----------

def test_webhook_app(env):
    bot, _, _, api, *_ = env
    client = TestClient(create_app(bot, SECRET))
    body = json.dumps({"events": [_event(message_type="sticker")]}).encode()

    response = client.post("/callback", content=body, headers={"X-Line-Signature": compute_signature(body, SECRET)})
    assert response.status_code == 200 and api.calls[-1][2] == [NON_TEXT_TEXT]

    assert client.post("/callback", content=body, headers={"X-Line-Signature": "wrong"}).status_code == 400
    assert client.post("/callback", content=body).status_code == 400
    bad = b"not json"
    assert client.post("/callback", content=bad, headers={"X-Line-Signature": compute_signature(bad, SECRET)}).status_code == 400
    assert client.get("/health").json() == {"status": "ok"}


# ---------- 啟動輔助 ----------

def test_find_cloudflared(tmp_path, monkeypatch):
    exe = tmp_path / "cloudflared.exe"
    exe.write_text("")
    assert find_cloudflared(str(exe)) == str(exe)
    with pytest.raises(LineSetupError, match="cloudflared_path"):
        find_cloudflared(str(tmp_path / "nope.exe"))

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/cloudflared")
    assert find_cloudflared("") == "/usr/bin/cloudflared"
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert find_cloudflared("", default=exe) == str(exe)
    with pytest.raises(LineSetupError, match="winget install"):
        find_cloudflared("", default=tmp_path / "missing.exe")


def test_read_secrets(tmp_path):
    env_path = tmp_path / ".env"
    assert read_secrets({"LINE_CHANNEL_SECRET": "s", "LINE_CHANNEL_ACCESS_TOKEN": "t"}, env_path) == ("s", "t")
    with pytest.raises(LineSetupError, match="LINE_CHANNEL_ACCESS_TOKEN"):
        read_secrets({"LINE_CHANNEL_SECRET": "s"}, env_path)


def test_wait_for_tunnel_url():
    lines = ["INF Starting tunnel", "INF |  https://abc-def-123.trycloudflare.com  |", "more"]
    assert wait_for_tunnel_url(iter(lines), timeout=10) == "https://abc-def-123.trycloudflare.com"
    with pytest.raises(LineSetupError, match="無法建立 Cloudflare 通道"):
        wait_for_tunnel_url(iter(["no url"]), timeout=10)
    ticks = itertools.count(0, 100)
    with pytest.raises(LineSetupError):
        wait_for_tunnel_url(iter(["a", "b", "https://x.trycloudflare.com"]), timeout=150, clock=lambda: next(ticks))


def test_wait_until_reachable():
    from rag_chatbot.line.runner import wait_until_reachable

    responses = iter([ConnectionError("dns"), 502, 200])

    def get(url):
        result = next(responses)
        if isinstance(result, Exception):
            raise result
        return result

    sleeps = []
    wait_until_reachable("https://x/health", get, timeout=60, clock=lambda: 0, sleep=sleeps.append)
    assert len(sleeps) == 2

    ticks = itertools.count(0, 50)
    with pytest.raises(LineSetupError, match="無法連線"):
        wait_until_reachable("https://x/health", lambda u: 502, timeout=100, clock=lambda: next(ticks), sleep=lambda s: None)


def test_retry():
    from rag_chatbot.line.runner import retry

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise LineApiError(400, "Invalid webhook endpoint URL")
        return "ok"

    sleeps = []
    assert retry(flaky, attempts=5, interval=5, sleep=sleeps.append) == "ok"
    assert sleeps == [5, 5]
    with pytest.raises(LineApiError):
        retry(lambda: (_ for _ in ()).throw(LineApiError(400, "x")), attempts=2, interval=1, sleep=lambda s: None)

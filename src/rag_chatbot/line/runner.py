"""啟動 LINE 機器人：webhook、Cloudflare 快速通道、自動更新 Webhook 網址（docs/05-line.md §4.4）。

python -X utf8 -m rag_chatbot line
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from rag_chatbot.config import Config

DEFAULT_CLOUDFLARED = Path(r"C:\Program Files (x86)\cloudflared\cloudflared.exe")
_TUNNEL_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
SECRET_KEY = "LINE_CHANNEL_SECRET"
TOKEN_KEY = "LINE_CHANNEL_ACCESS_TOKEN"


class LineSetupError(Exception):
    """啟動前的準備不完整（金鑰、cloudflared、通道等）。"""


def find_cloudflared(configured: str, default: Path = DEFAULT_CLOUDFLARED) -> str:
    """設定值 → PATH → 預設安裝位置。"""
    if configured:
        if Path(configured).is_file():
            return configured
        raise LineSetupError(f"找不到 cloudflared：{configured}（[line] cloudflared_path）")
    found = shutil.which("cloudflared")
    if found:
        return found
    if default.is_file():
        return str(default)
    raise LineSetupError("找不到 cloudflared，請先執行：winget install Cloudflare.cloudflared")


def read_secrets(env: dict[str, str], env_path: Path) -> tuple[str, str]:
    missing = [k for k in (SECRET_KEY, TOKEN_KEY) if not env.get(k)]
    if missing:
        raise LineSetupError(
            f"{env_path} 缺少 {', '.join(missing)}。請用記事本建立此檔，內容為：\n"
            f"  {SECRET_KEY}=（Channel secret）\n  {TOKEN_KEY}=（Channel access token）"
        )
    return env[SECRET_KEY], env[TOKEN_KEY]


def wait_for_tunnel_url(lines: Iterable[str], timeout: float, clock: Callable[[], float] = time.monotonic) -> str:
    """從 cloudflared 的輸出中找出 https://xxxx.trycloudflare.com。"""
    deadline = clock() + timeout
    for line in lines:
        match = _TUNNEL_URL.search(line)
        if match:
            return match.group(0)
        if clock() > deadline:
            break
    raise LineSetupError("無法建立 Cloudflare 通道（未取得網址），請確認網路連線後重試")


def wait_until_reachable(
    url: str,
    get: Callable[[str], int],
    timeout: float,
    interval: float = 2.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """新的通道網址需數秒到數十秒 DNS 才生效；LINE 設定 Webhook 時若連不到會回「Invalid webhook endpoint URL」。

    get(網址) 回傳 HTTP 狀態碼，連線失敗時丟出例外。
    """
    deadline = clock() + timeout
    while True:
        try:
            if get(url) == 200:
                return
        except Exception:  # 連線失敗、DNS 尚未生效
            pass
        if clock() >= deadline:
            raise LineSetupError(f"通道網址在 {timeout:.0f} 秒內無法連線：{url}")
        sleep(interval)


def retry(action: Callable[[], object], attempts: int, interval: float, sleep: Callable[[float], None] = time.sleep):
    """失敗時重試；最後一次的錯誤直接丟出。"""
    for attempt in range(attempts):
        try:
            return action()
        except Exception:
            if attempt == attempts - 1:
                raise
            sleep(interval)


def run(config: Config, env_path: Path) -> int:  # pragma: no cover - 以功能測試 FT-23～30 驗證
    """啟動並持續執行，Ctrl+C 停止。"""
    import uvicorn

    from rag_chatbot.db import connect
    from rag_chatbot.embedder import OllamaEmbedder
    from rag_chatbot.env import load_env
    from rag_chatbot.line.api import HttpLineApi
    from rag_chatbot.line.app import create_app
    from rag_chatbot.line.handler import LineBot
    from rag_chatbot.llm import OllamaLLM
    from rag_chatbot.profiles import load_profile
    from rag_chatbot.rag import RagService
    from rag_chatbot.vector_store import VectorStore

    secret, token = read_secrets(load_env(env_path), env_path)
    profile = load_profile(config, config.line.profile)
    cloudflared = find_cloudflared(config.line.cloudflared_path)

    print("檢查 Ollama 並載入模型…")
    OllamaEmbedder.from_config(config).embed(["測試"])
    OllamaLLM.from_config(config).chat([{"role": "user", "content": "請回答：好"}])

    def make_service() -> RagService:
        store = VectorStore(connect(config.paths.db), config.ollama.embed_model)
        return RagService(config, store, OllamaEmbedder.from_config(config), OllamaLLM.from_config(config))

    api = HttpLineApi(token)
    bot = LineBot(config, api)
    stop = threading.Event()
    threading.Thread(target=bot.run_worker, args=(make_service, stop), daemon=True).start()

    port = config.line.port
    server = uvicorn.Server(uvicorn.Config(create_app(bot, secret), host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.1)

    print("建立 Cloudflare 通道…")
    tunnel = subprocess.Popen(
        [cloudflared, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
    )
    try:
        url = wait_for_tunnel_url(tunnel.stderr, timeout=60)
        # 之後持續讀取輸出，避免管線塞滿讓 cloudflared 停住
        threading.Thread(target=lambda: [None for _ in tunnel.stderr], daemon=True).start()

        print(f"等待通道網址生效：{url}")
        import httpx

        wait_until_reachable(f"{url}/health", lambda u: httpx.get(u, timeout=5).status_code, timeout=120)
        endpoint = f"{url}/callback"
        retry(lambda: api.set_webhook(endpoint), attempts=6, interval=5)
        result: dict = {}
        for _ in range(12):  # 新網址生效需要數秒
            result = api.test_webhook()
            if result.get("success"):
                break
            time.sleep(5)
        info = api.webhook_info()

        print(f"\nLINE 機器人已上線（情境 {profile.name}）")
        print(f"Webhook 網址：{endpoint}")
        print(f"連線測試：{'成功' if result.get('success') else '失敗：' + str(result.get('reason') or result)}")
        if not info.get("active"):
            print("注意：LINE 的「Use webhook」尚未開啟。請到 LINE Developers Console →"
                  "「Messaging API」→「Webhook settings」打開「Use webhook」。")
        print("按 Ctrl+C 停止。\n")
        while tunnel.poll() is None:
            time.sleep(1)
        print("Cloudflare 通道已中斷，請重新啟動。")
        return 1
    except KeyboardInterrupt:
        print("\n停止中…")
        return 0
    finally:
        stop.set()
        server.should_exit = True
        tunnel.terminate()

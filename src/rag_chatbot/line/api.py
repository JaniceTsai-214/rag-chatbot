"""LINE Messaging API（只實作需要的 6 個端點，見 docs/05-line.md §6）。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import httpx

API_BASE = "https://api.line.me/v2/bot"
MAX_TEXT_CHARS = 5000  # LINE 單則文字訊息上限
MAX_MESSAGES = 5  # 單次 reply / push 最多則數
LOADING_SECONDS = 60  # 「輸入中」動畫最長時間


class LineApiError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"LINE API 錯誤 {status}：{message}")
        self.status = status


class LineApi(Protocol):
    def reply(self, reply_token: str, texts: Sequence[str]) -> None: ...
    def push(self, user_id: str, texts: Sequence[str]) -> None: ...
    def start_loading(self, user_id: str, seconds: int = LOADING_SECONDS) -> None: ...


class HttpLineApi:
    def __init__(self, access_token: str, client: httpx.Client | None = None, timeout: float = 10.0) -> None:
        self._client = client or httpx.Client(timeout=timeout)
        self._headers = {"Authorization": f"Bearer {access_token}"}

    def reply(self, reply_token: str, texts: Sequence[str]) -> None:
        self._post("/message/reply", {"replyToken": reply_token, "messages": _messages(texts)})

    def push(self, user_id: str, texts: Sequence[str]) -> None:
        self._post("/message/push", {"to": user_id, "messages": _messages(texts)})

    def start_loading(self, user_id: str, seconds: int = LOADING_SECONDS) -> None:
        self._post("/chat/loading/start", {"chatId": user_id, "loadingSeconds": seconds})

    def set_webhook(self, endpoint: str) -> None:
        self._request("PUT", "/channel/webhook/endpoint", {"endpoint": endpoint})

    def webhook_info(self) -> dict[str, Any]:
        """{"endpoint": 網址, "active": 是否已開啟 Use webhook}"""
        return self._request("GET", "/channel/webhook/endpoint")

    def test_webhook(self) -> dict[str, Any]:
        """請 LINE 對目前的 Webhook 網址送測試請求；{"success": bool, "statusCode": ..., "reason": ...}"""
        return self._request("POST", "/channel/webhook/test", {})

    def _post(self, path: str, payload: dict[str, Any]) -> None:
        self._request("POST", path, payload)

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            response = self._client.request(method, API_BASE + path, headers=self._headers, json=payload)
        except httpx.HTTPError as e:
            raise LineApiError(0, f"無法連線到 LINE（{e}）") from e
        if response.status_code >= 400:
            try:
                message = response.json().get("message", response.text)
            except ValueError:
                message = response.text
            raise LineApiError(response.status_code, message)
        return response.json() if response.content else {}


def _messages(texts: Sequence[str]) -> list[dict[str, str]]:
    return [{"type": "text", "text": t[:MAX_TEXT_CHARS]} for t in texts[:MAX_MESSAGES]]

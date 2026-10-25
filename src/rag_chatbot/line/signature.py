"""LINE webhook 簽章驗證：X-Line-Signature = Base64(HMAC-SHA256(channel secret, 請求內容))。"""

from __future__ import annotations

import base64
import hashlib
import hmac


def compute_signature(body: bytes, channel_secret: str) -> str:
    digest = hmac.new(channel_secret.encode("utf-8"), body, hashlib.sha256).digest()
    return base64.b64encode(digest).decode("ascii")


def verify_signature(body: bytes, signature: str | None, channel_secret: str) -> bool:
    if not signature:
        return False
    return hmac.compare_digest(compute_signature(body, channel_secret), signature)

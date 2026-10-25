"""問候、閒聊、套話判斷（設計見 docs/02-design.md §4.2 步驟 2，D-023）。

規則無法涵蓋所有說法；漏網的由 Prompt 規則處理。
"""

from __future__ import annotations

import re
from enum import Enum


class SmallTalk(Enum):
    GREETING = "greeting"  # 問候、詢問身分
    THANKS = "thanks"  # 道謝、「好的」等回應
    FAREWELL = "farewell"  # 道別
    INJECTION = "injection"  # 套話：要求忽略規則、索取系統提示


MAX_SHORT_CHARS = 12  # 去除空白與標點後的長度上限；超過視為一般問題

_PARTICLES = "(?:啊|呀|喔|哦|啦|嗎|囉|耶|唷|欸|～|你|妳|您|大家)*"


def _phrases(*words: str) -> str:
    return "(?:" + "|".join(sorted(words, key=len, reverse=True)) + ")"


_GREETING = _phrases(
    "你好", "妳好", "您好", "哈囉", "嗨", "安安", "早安", "午安", "晚安", "早",
    "hi", "hello", "hey", "在嗎", "有人嗎", "有人在嗎",
    "你是誰", "妳是誰", "您是誰", "你是什麼", "妳是什麼", "你叫什麼", "你叫什麼名字",
    "whoareyou",
)
_THANKS = _phrases(
    "謝謝", "謝啦", "感謝", "多謝", "感恩", "thanks", "thankyou", "thx",
    "好的", "好", "了解", "瞭解", "收到", "知道了", "ok", "okay",
)
_FAREWELL = _phrases("掰掰", "拜拜", "再見", "bye", "byebye", "goodbye", "下次見")

_SHORT_PATTERNS = (
    (SmallTalk.FAREWELL, re.compile(f"(?:{_FAREWELL}{_PARTICLES})+")),
    (SmallTalk.THANKS, re.compile(f"(?:{_THANKS}{_PARTICLES})+")),
    (SmallTalk.GREETING, re.compile(f"(?:(?:{_GREETING}|{_THANKS}|{_FAREWELL}){_PARTICLES})+")),
)

_INJECTION = re.compile(
    "|".join([
        r"(忽略|無視|忘記|忘掉|不要理會|跳過).{0,8}(規則|指示|指令|設定|提示|限制)",
        r"(系統|系统)(提示|指令|設定)",
        r"提示詞",
        r"system\s*prompt",
        r"(ignore|disregard|forget)\s+(all\s+|the\s+|any\s+|your\s+)?(previous|prior|above|earlier)?\s*"
        r"(instructions?|rules?|prompts?)",
        r"developer\s*mode|開發者模式|jailbreak|越獄",
    ]),
    re.IGNORECASE,
)


def classify(message: str) -> SmallTalk | None:
    """判斷訊息是否為問候、閒聊或套話；一般問題回傳 None。"""
    if _INJECTION.search(message):
        return SmallTalk.INJECTION
    normalized = re.sub(r"[\W_]+", "", message.lower())
    if not normalized or len(normalized) > MAX_SHORT_CHARS:
        return None
    for kind, pattern in _SHORT_PATTERNS:
        if pattern.fullmatch(normalized):
            return kind
    return None

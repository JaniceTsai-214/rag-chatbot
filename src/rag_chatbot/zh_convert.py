"""簡體字轉繁體（D-027）：qwen2.5 偶爾輸出簡體字，回答送出前轉換。

只轉換「含有簡體字的句子」：Big5（cp950）沒有的中文字視為簡體字。
不整段轉換，因為繁簡共用的字（如「干擾」的「干」、「台灣」的「台」）在繁體文字中會被誤轉。
使用 OpenCC s2tw（只轉字形，不換用語），避免「天行智能」被改成「天行智慧」之類的專有名詞變動。
"""

from __future__ import annotations

import re
from functools import lru_cache

from opencc import OpenCC

_CJK = re.compile(r"[㐀-䶿一-鿿]")
_SENTENCE = re.compile(r"[^。！？!?\n]*[。！？!?\n]?")


@lru_cache(maxsize=1)
def _converter() -> OpenCC:
    return OpenCC("s2tw")


def _is_simplified_only(char: str) -> bool:
    try:
        char.encode("cp950")
    except UnicodeEncodeError:
        return True
    return False


def has_simplified(text: str) -> bool:
    return any(_is_simplified_only(c) for c in _CJK.findall(text))


def to_traditional(text: str) -> str:
    """把含有簡體字的句子轉成繁體；其他句子原樣保留。"""
    if not has_simplified(text):
        return text
    converter = _converter()
    return "".join(
        converter.convert(sentence) if has_simplified(sentence) else sentence
        for sentence in _SENTENCE.findall(text)
    )

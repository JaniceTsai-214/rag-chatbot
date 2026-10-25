"""關鍵字檢索：BM25，中文取相鄰兩字（bigram）、英數字依單字（設計見 docs/02-design.md §4.2 步驟 4）。"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence

import numpy as np

_WORD = re.compile(r"[a-z0-9]+")
_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]+")

# BM25 參數（常用預設值）
_K1 = 1.5
_B = 0.75


def tokenize(text: str) -> list[str]:
    """英數字轉小寫後依單字切開；連續中文取相鄰兩字，單一中文字保留原字。"""
    text = text.lower()
    tokens = _WORD.findall(text)
    for run in _CJK.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


class KeywordIndex:
    """對一組文件建立 BM25 索引；scores() 回傳與輸入順序一致的分數陣列。"""

    def __init__(self, documents: Sequence[str]) -> None:
        self.size = len(documents)
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths = np.zeros(self.size, dtype=np.float64)
        for i, doc in enumerate(documents):
            counts = Counter(tokenize(doc))
            lengths[i] = sum(counts.values())
            for token, tf in counts.items():
                self._postings[token].append((i, tf))
        average = lengths.mean() if self.size and lengths.mean() > 0 else 1.0
        self._norm = _K1 * (1 - _B + _B * lengths / average)

    def scores(self, query: str) -> np.ndarray:
        result = np.zeros(self.size, dtype=np.float64)
        for token in set(tokenize(query)):
            postings = self._postings.get(token)
            if not postings:
                continue
            df = len(postings)
            idf = math.log(1 + (self.size - df + 0.5) / (df + 0.5))
            for i, tf in postings:
                result[i] += idf * tf * (_K1 + 1) / (tf + self._norm[i])
        return result

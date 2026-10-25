"""讀取情境設定 profiles/*.toml（設計見 docs/02-design.md §3）。"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from rag_chatbot.config import Config

_REQUIRED = ("title", "subject", "collections", "contact", "refusal_text", "system_prompt")
_OPTIONAL = ("top_k", "min_score", "greeting_text", "thanks_text", "farewell_text")

DEFAULT_GREETING = "您好，我是{subject}的{title}，歡迎詢問相關問題。"
DEFAULT_THANKS = "不客氣！還有其他想了解的嗎？"
DEFAULT_FAREWELL = "謝謝您的詢問，祝您有美好的一天！"


class ProfileError(Exception):
    """情境設定錯誤。"""


@dataclass(frozen=True)
class Profile:
    name: str
    title: str
    subject: str
    collections: tuple[str, ...]
    contact: str
    refusal_text_template: str
    system_prompt: str  # 預設 Prompt 範本（含變數），首次使用時存為 v1
    top_k: int | None = None
    min_score: float | None = None
    greeting_text_template: str = DEFAULT_GREETING
    thanks_text_template: str = DEFAULT_THANKS
    farewell_text_template: str = DEFAULT_FAREWELL

    @property
    def refusal_text(self) -> str:
        """代入變數後的拒答文字。"""
        return self._fill(self.refusal_text_template)

    @property
    def greeting_text(self) -> str:
        """問候、詢問身分、套話時的回覆。"""
        return self._fill(self.greeting_text_template)

    @property
    def thanks_text(self) -> str:
        return self._fill(self.thanks_text_template)

    @property
    def farewell_text(self) -> str:
        return self._fill(self.farewell_text_template)

    def _fill(self, template: str) -> str:
        return _substitute(template, subject=self.subject, contact=self.contact, title=self.title)

    def render_prompt(self, template: str | None = None) -> str:
        """將 Prompt 範本代入變數；未指定範本時使用情境預設 Prompt。"""
        return _substitute(
            self.system_prompt if template is None else template,
            subject=self.subject,
            contact=self.contact,
            refusal_text=self.refusal_text,
        )

    def resolved_top_k(self, config: Config) -> int:
        return self.top_k if self.top_k is not None else config.retrieval.top_k

    def resolved_min_score(self, config: Config) -> float:
        return self.min_score if self.min_score is not None else config.retrieval.min_score

    def missing_collections(self, config: Config) -> list[str]:
        """列出資料夾不存在的文件集。"""
        return [c for c in self.collections if not (config.paths.collections / c).is_dir()]


def _substitute(template: str, **values: str) -> str:
    # 只取代已知變數；Prompt 中其他大括號原樣保留
    for key, value in values.items():
        template = template.replace("{" + key + "}", value)
    return template


def list_profiles(config: Config) -> list[Profile]:
    """依名稱排序列出所有情境。"""
    folder = config.paths.profiles
    if not folder.is_dir():
        raise ProfileError(f"找不到情境資料夾：{folder}")
    return [_load_file(p) for p in sorted(folder.glob("*.toml"))]


def load_profile(config: Config, name: str | None = None) -> Profile:
    """讀取指定情境；未指定時使用 config 的預設情境。"""
    name = name or config.default_profile
    path = config.paths.profiles / f"{name}.toml"
    if not path.is_file():
        raise ProfileError(f"找不到情境「{name}」：{path}")
    return _load_file(path)


def _load_file(path: Path) -> Profile:
    name = path.stem
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ProfileError(f"情境「{name}」格式錯誤：{e}") from e

    missing = [k for k in _REQUIRED if k not in data]
    if missing:
        raise ProfileError(f"情境「{name}」缺少項目：{', '.join(missing)}")
    unknown = set(data) - set(_REQUIRED) - set(_OPTIONAL)
    if unknown:
        raise ProfileError(f"情境「{name}」有未知的項目：{', '.join(sorted(unknown))}")

    for key in ("title", "subject", "contact", "refusal_text", "system_prompt",
                "greeting_text", "thanks_text", "farewell_text"):
        if key in data and (not isinstance(data[key], str) or not data[key].strip()):
            raise ProfileError(f"情境「{name}」的 {key} 必須是非空字串")

    collections = data["collections"]
    if (
        not isinstance(collections, list)
        or not collections
        or not all(isinstance(c, str) and c for c in collections)
    ):
        raise ProfileError(f"情境「{name}」的 collections 必須是非空的字串清單")
    for c in collections:
        if "/" in c or "\\" in c or c in (".", ".."):
            raise ProfileError(f"情境「{name}」的文件集名稱不合法：{c}")

    top_k = data.get("top_k")
    if top_k is not None and (not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0):
        raise ProfileError(f"情境「{name}」的 top_k 必須是正整數")
    min_score = data.get("min_score")
    if min_score is not None:
        if not isinstance(min_score, (int, float)) or isinstance(min_score, bool):
            raise ProfileError(f"情境「{name}」的 min_score 必須是數字")
        if not 0.0 <= min_score <= 1.0:
            raise ProfileError(f"情境「{name}」的 min_score 必須介於 0 與 1 之間")
        min_score = float(min_score)

    return Profile(
        name=name,
        title=data["title"],
        subject=data["subject"],
        collections=tuple(collections),
        contact=data["contact"],
        refusal_text_template=data["refusal_text"],
        system_prompt=data["system_prompt"],
        top_k=top_k,
        min_score=min_score,
        greeting_text_template=data.get("greeting_text", DEFAULT_GREETING).strip(),
        thanks_text_template=data.get("thanks_text", DEFAULT_THANKS).strip(),
        farewell_text_template=data.get("farewell_text", DEFAULT_FAREWELL).strip(),
    )

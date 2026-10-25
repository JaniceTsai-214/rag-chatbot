"""讀取系統設定檔 config.toml（設計見 docs/02-design.md）。"""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path("config.toml")


class ConfigError(Exception):
    """設定檔內容錯誤。"""


@dataclass(frozen=True)
class OllamaConfig:
    host: str = "http://127.0.0.1:11434"
    chat_model: str = "qwen2.5:3b"
    embed_model: str = "bge-m3"
    timeout_seconds: int = 300
    chat_num_ctx: int = 4096
    chat_max_tokens: int = 300
    chat_temperature: float = 0.2
    keep_alive: str = "30m"


@dataclass(frozen=True)
class PathsConfig:
    db: Path = Path("data/app.db")
    collections: Path = Path("data/collections")
    profiles: Path = Path("profiles")


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = 600
    chunk_overlap: int = 100


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 5
    min_score: float = 0.40
    history_turns: int = 2
    max_context_chars: int = 1800


@dataclass(frozen=True)
class LineConfig:
    profile: str = "intro"
    port: int = 8000
    reply_deadline_seconds: int = 50
    conversation_idle_minutes: int = 30
    rate_limit_count: int = 10
    rate_limit_minutes: int = 10
    max_queue: int = 3
    cloudflared_path: str = ""


@dataclass(frozen=True)
class Config:
    default_profile: str = "intro"
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    line: LineConfig = field(default_factory=LineConfig)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """讀取設定檔。未寫的項目使用預設值；相對路徑以設定檔所在資料夾為準。"""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"找不到設定檔：{path}")
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"設定檔格式錯誤（{path}）：{e}") from e

    sections = {
        "ollama": OllamaConfig,
        "paths": PathsConfig,
        "chunking": ChunkingConfig,
        "retrieval": RetrievalConfig,
        "line": LineConfig,
    }
    unknown = set(data) - set(sections) - {"default_profile"}
    if unknown:
        raise ConfigError(f"設定檔有未知的項目：{', '.join(sorted(unknown))}")

    parsed = {name: _build(cls, data.get(name, {}), name) for name, cls in sections.items()}
    base = path.resolve().parent
    paths: PathsConfig = parsed["paths"]
    parsed["paths"] = PathsConfig(
        db=base / paths.db,
        collections=base / paths.collections,
        profiles=base / paths.profiles,
    )
    default_profile = data.get("default_profile", Config.default_profile)
    if not isinstance(default_profile, str) or not default_profile:
        raise ConfigError("default_profile 必須是非空字串")

    config = Config(default_profile=default_profile, **parsed)
    _validate(config)
    return config


def _build(cls: type, values: Any, section: str) -> Any:
    """依 dataclass 欄位型別建立設定區塊，檢查未知項目與型別。"""
    if not isinstance(values, dict):
        raise ConfigError(f"[{section}] 必須是區塊")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(values) - set(fields)
    if unknown:
        raise ConfigError(f"[{section}] 有未知的項目：{', '.join(sorted(unknown))}")

    kwargs = {}
    for name, value in values.items():
        expected = type(getattr(cls(), name))
        if issubclass(expected, Path):
            if not isinstance(value, str):
                raise ConfigError(f"[{section}] {name} 必須是字串路徑")
            value = Path(value)
        elif expected is float and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        elif not isinstance(value, expected) or isinstance(value, bool):
            raise ConfigError(f"[{section}] {name} 型別錯誤，應為 {expected.__name__}")
        kwargs[name] = value
    return cls(**kwargs)


def _validate(config: Config) -> None:
    c, r = config.chunking, config.retrieval
    if c.chunk_size <= 0:
        raise ConfigError("[chunking] chunk_size 必須大於 0")
    if not 0 <= c.chunk_overlap < c.chunk_size:
        raise ConfigError("[chunking] chunk_overlap 必須介於 0 與 chunk_size 之間")
    if r.top_k <= 0:
        raise ConfigError("[retrieval] top_k 必須大於 0")
    if not 0.0 <= r.min_score <= 1.0:
        raise ConfigError("[retrieval] min_score 必須介於 0 與 1 之間")
    if r.history_turns < 0:
        raise ConfigError("[retrieval] history_turns 不可為負數")
    if r.max_context_chars <= 0:
        raise ConfigError("[retrieval] max_context_chars 必須大於 0")
    o = config.ollama
    if o.timeout_seconds <= 0:
        raise ConfigError("[ollama] timeout_seconds 必須大於 0")
    if o.chat_num_ctx <= 0 or o.chat_max_tokens <= 0:
        raise ConfigError("[ollama] chat_num_ctx、chat_max_tokens 必須大於 0")
    if not 0.0 <= o.chat_temperature <= 2.0:
        raise ConfigError("[ollama] chat_temperature 必須介於 0 與 2 之間")
    line = config.line
    if not 1 <= line.port <= 65535:
        raise ConfigError("[line] port 必須介於 1 與 65535 之間")
    for name in ("reply_deadline_seconds", "conversation_idle_minutes", "rate_limit_count",
                 "rate_limit_minutes", "max_queue"):
        if getattr(line, name) <= 0:
            raise ConfigError(f"[line] {name} 必須大於 0")

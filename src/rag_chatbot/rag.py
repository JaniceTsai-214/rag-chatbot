"""answer() 核心：問候判斷 → 混合檢索 → 門檻 → 組 Prompt → LLM → 後處理 → 對話紀錄
（設計見 docs/02-design.md §4.2）。管理介面、命令列與 LINE 共用。
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from pathlib import PurePath

from rag_chatbot.config import Config
from rag_chatbot.embedder import Embedder
from rag_chatbot.history import ASSISTANT, USER, HistoryError, HistoryStore
from rag_chatbot.llm import LLM, Message
from rag_chatbot.models import ANSWERED, GREETING, REFUSED, Answer, Source
from rag_chatbot.profiles import Profile, load_profile
from rag_chatbot.prompts import PromptStore
from rag_chatbot.retriever import Retriever
from rag_chatbot.smalltalk import SmallTalk, classify
from rag_chatbot.vector_store import VectorStore
from rag_chatbot.zh_convert import to_traditional

ERROR = "error"  # 只用於對話紀錄：呼叫 Ollama 失敗


class RagService:
    def __init__(
        self,
        config: Config,
        store: VectorStore,
        embedder: Embedder,
        llm: LLM,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.config = config
        self.llm = llm
        self.retriever = Retriever(store, embedder)
        self.prompts = PromptStore(store.conn)
        self.history = HistoryStore(store.conn)
        self._clock = clock

    def answer(
        self,
        question: str,
        profile: str | None = None,
        conversation_id: int | None = None,
        prompt_version: int | None = None,
    ) -> Answer:
        """回答問題並寫入對話紀錄。profile 為 None 時使用預設情境；conversation_id 為 None 時建立新對話。"""
        start = self._clock()
        question = question.strip()
        if not question:
            raise ValueError("問題不可為空白")
        p = load_profile(self.config, profile)
        conversation_id = self._conversation(p, conversation_id, question)
        previous = [
            q for q in self.history.recent_questions(conversation_id, self.config.retrieval.history_turns)
            if classify(q) is None
        ]
        self.history.add_message(conversation_id, USER, question)

        def finish(kind: str, text: str, **fields) -> Answer:
            latency_ms = int((self._clock() - start) * 1000)
            self.history.add_message(conversation_id, ASSISTANT, text, kind=kind, latency_ms=latency_ms, **fields)
            return Answer(
                kind=kind, text=text, conversation_id=conversation_id, latency_ms=latency_ms,
                sources=list(fields.get("sources", [])), model=fields.get("model"),
                prompt_version=fields.get("prompt_version"), top_score=fields.get("top_score"),
            )

        talk = classify(question)
        if talk is not None:
            return finish(GREETING, _smalltalk_reply(p, talk))

        try:
            prompt = self.prompts.resolve(p, prompt_version)
            result = self.retriever.retrieve(
                "\n".join([*previous, question]),
                p.collections,
                p.resolved_top_k(self.config),
                p.resolved_min_score(self.config),
                self.config.retrieval.max_context_chars,
            )
            if not result.passed:
                return finish(REFUSED, p.refusal_text, top_score=result.top_score)

            raw = self.llm.chat(build_messages(p.render_prompt(prompt.content), result.sources, previous, question))
        except Exception as e:
            # 問題已寫入紀錄；補上一筆錯誤訊息，方便事後檢視，再把錯誤交給呼叫端
            self.history.add_message(conversation_id, ASSISTANT, str(e), kind=ERROR)
            raise

        kind, text = postprocess(raw, p.refusal_text)
        return finish(
            kind, text, sources=result.sources, model=self.llm.model,
            prompt_version=prompt.version, top_score=result.top_score,
        )

    def _conversation(self, profile: Profile, conversation_id: int | None, question: str) -> int:
        if conversation_id is None:
            return self.history.create_conversation(profile.name, question).id
        conversation = self.history.get_conversation(conversation_id)
        if conversation is None:
            raise HistoryError(f"對話 {conversation_id} 不存在")
        if conversation.profile != profile.name:
            raise HistoryError(f"對話 {conversation_id} 屬於情境「{conversation.profile}」，不是「{profile.name}」")
        return conversation_id


def _smalltalk_reply(profile: Profile, talk: SmallTalk) -> str:
    if talk is SmallTalk.THANKS:
        return profile.thanks_text
    if talk is SmallTalk.FAREWELL:
        return profile.farewell_text
    return profile.greeting_text  # 問候、詢問身分、套話


def source_label(source: Source) -> str:
    """參考資料的出處，例如「蔡佳惠履歷 > 學歷」「智慧訂單簡報｜第 4 頁」。"""
    label = " > ".join(p for p in (PurePath(source.filename).stem, source.heading) if p)
    return f"{label}｜第 {source.page} 頁" if source.page else label


def build_messages(
    system_prompt: str,
    sources: Sequence[Source],
    previous: Sequence[str],
    question: str,
) -> list[Message]:
    """system：情境 Prompt；user：編號參考資料（含出處）+ 先前的問題 + 本次問題。"""
    parts = ["參考資料："]
    parts += [f"[{i}]【{source_label(s)}】\n{s.text}" for i, s in enumerate(sources, start=1)]
    if previous:
        parts.append("先前的問題：\n" + "\n".join(f"- {q}" for q in previous))
    parts.append(f"問題：{question}")
    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


_SENTENCE_END = "。！？!?\n"


def postprocess(raw: str, refusal_text: str) -> tuple[str, str]:
    """拒答句的後處理（D-022）：
    - 只有拒答句，或回答一開頭就是拒答句（模型常改寫後半句再接著發揮）→ 拒答，改回標準拒答文字。
    - 有內容又附上拒答句 → 刪除該句。
    拒答句以完整文字比對，也以開頭片段（第一個「，」或「。」之前）比對改寫過的版本。
    處理前先把含簡體字的句子轉成繁體（D-027）。
    """
    text = to_traditional(raw.strip())
    text = text.replace(f"「{refusal_text}」", refusal_text)
    core = _refusal_core(refusal_text)
    position = text.find(core)
    if position == -1:
        return ANSWERED, text
    if not _has_words(text[:position]):
        return REFUSED, refusal_text

    remainder = text.replace(refusal_text, "")
    while (position := remainder.find(core)) != -1:  # 改寫過的拒答句：刪除所在的整句
        start = max(remainder.rfind(c, 0, position) for c in _SENTENCE_END) + 1
        ends = [i for c in _SENTENCE_END if (i := remainder.find(c, position)) != -1]
        end = min(ends) + 1 if ends else len(remainder)
        remainder = remainder[:start] + remainder[end:]
    remainder = re.sub(r"\n{3,}", "\n\n", remainder).strip()
    return (ANSWERED, remainder) if _has_words(remainder) else (REFUSED, refusal_text)


def _refusal_core(refusal_text: str) -> str:
    core = re.split(r"[，。,]", refusal_text, maxsplit=1)[0].strip()
    return core if len(core) >= 4 else refusal_text


def _has_words(text: str) -> bool:
    return bool(re.sub(r"[\W_]+", "", text))

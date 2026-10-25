"""FastAPI webhook：POST /callback（docs/05-line.md §4.1）。"""

from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from rag_chatbot.line.handler import LineBot
from rag_chatbot.line.signature import verify_signature


def create_app(bot: LineBot, channel_secret: str) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.post("/callback")
    async def callback(request: Request) -> dict[str, str]:
        body = await request.body()
        if not verify_signature(body, request.headers.get("x-line-signature"), channel_secret):
            raise HTTPException(status_code=400, detail="Invalid signature")
        try:
            payload = json.loads(body)
        except ValueError as e:
            raise HTTPException(status_code=400, detail="Invalid JSON") from e
        # accept() 可能直接呼叫 LINE API 回覆（貼圖、頻率限制），放到執行緒避免阻塞
        await run_in_threadpool(bot.accept, payload)
        return {"status": "ok"}

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app

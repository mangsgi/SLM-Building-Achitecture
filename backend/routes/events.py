# routes/events.py
import json, asyncio
from typing import AsyncIterator
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from deps import rds, train_event_channel

router = APIRouter()

async def _sse(data: dict) -> bytes:
    # SSE payload는 한 이벤트를 한 덩어리로 전송
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")

async def _retry(ms: int = 5000) -> bytes:
    # 브라우저 자동 재연결 간격(밀리초)
    return f"retry: {ms}\n\n".encode("utf-8")

async def _hb() -> bytes:
    # 주석 라인(하트비트)
    return b": keep-alive\n\n"

async def _stream(channel: str) -> AsyncIterator[bytes]:
    pubsub = rds.pubsub()
    await pubsub.subscribe(channel)
    try:
        # 재시도 간격 먼저 알림
        yield await _retry(5000)
        # 최초 연결 알림
        yield await _sse({"event": "connected", "data": {"channel": channel}})
        last = asyncio.get_event_loop().time()
        while True:
            msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            now = asyncio.get_event_loop().time()

            if msg and msg.get("type") == "message":
                raw = msg["data"]
                try:
                    data = json.loads(raw)
                except Exception:
                    data = {"event": "raw", "data": {"payload": raw}}
                yield await _sse(data)
                last = now

            # 15초마다 하트비트
            if now - last > 15:
                yield await _hb()
                last = now
    finally:
        try:
            await pubsub.unsubscribe(channel)
        finally:
            await pubsub.close()

# ★ 여기에서 /api/v1 붙이지 말 것 (main.py prefix 사용)
@router.get("/events/{task_id}")
async def events(task_id: str, request: Request):
    channel = train_event_channel(task_id)
    # 권장 헤더들 추가 (프록시/브라우저 안정성)
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",     # nginx 등에서 버퍼링 방지
        "Access-Control-Allow-Origin": "*",  # CORS 필요시
    }
    return StreamingResponse(_stream(channel), media_type="text/event-stream", headers=headers)

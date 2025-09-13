# routes/events.py
import json, asyncio
from typing import AsyncIterator
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from deps import rds, train_event_channel

# Celery 작업 상태 조회를 위해 추가
from celery.result import AsyncResult
from celery_app import celery_app

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

async def _stream(task_id: str, channel: str) -> AsyncIterator[bytes]:
    # 1. 연결 시작 시, 작업의 최종 상태를 먼저 확인 (Race Condition 방지) 
    # 프론트엔드가 SSE 연결 시 작업의 최종 상태를 먼저 확인하여 실패/성공 이벤트를 먼저 전송
    task = AsyncResult(task_id, app=celery_app)

    # 작업이 이미 실패한 경우
    if task.state == 'FAILURE':
        yield await _sse({"event": "error", "data": {"message": str(task.info)}})
        return

    # 작업이 이미 성공한 경우
    if task.state == 'SUCCESS':
        result = task.result
        # 결과 내용에 따라 성공/오류 이벤트 분기
        if isinstance(result, dict) and result.get('status') == 'error':
            yield await _sse({"event": "error", "data": {"message": result.get('message', 'Unknown error')}})
        else:
            yield await _sse({"event": "finished", "data": result})
        return

    # 2. 작업이 아직 실행 중이면, 실시간 이벤트 스트림 시작
    pubsub = rds.pubsub()
    await pubsub.subscribe(channel)
    try:
        # 재시도 간격 먼저 알림
        yield await _retry(5000)
        # 최초 연결 알림
        yield await _sse({"event": "connected", "data": {"channel": channel}})
        last = asyncio.get_event_loop().time()
        while True:
            # 작업이 끝났는지 다시 확인
            if AsyncResult(task_id, app=celery_app).ready():
                break

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
    return StreamingResponse(_stream(task_id, channel), media_type="text/event-stream", headers=headers)


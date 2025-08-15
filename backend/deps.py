# deps.py
import os
import redis                       
import redis.asyncio as aioredis   # a

# 환경변수 / 기본값
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# FastAPI(SSE)용 async redis 클라이언트
rds = aioredis.from_url(REDIS_URL, decode_responses=True)

# Celery 태스크용 sync redis 클라이언트
rds_sync = redis.from_url(REDIS_URL, decode_responses=True)

def train_event_channel(task_id: str) -> str:
    return f"task:{task_id}"

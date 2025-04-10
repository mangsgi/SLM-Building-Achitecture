from fastapi import APIRouter
from pydantic import BaseModel
from celery.result import AsyncResult


train_router = APIRouter()

# 요청 모델 정의
class TrainRequest(BaseModel):
    config: dict
    layer_config: dict  # 수정된 부분

# 모델 구성 요청
@train_router.post("/train")
def train_model(request: TrainRequest):
    from tasks import train_gpt_model
    task = train_gpt_model.apply_async(args=[request.layer_config, request.config])  # 수정된 부분
    return {"task_id": task.id}

# 결과 확인
@train_router.get("/train/{task_id}")
def get_result(task_id: str):
    task = AsyncResult(task_id)
    if task.ready():
        return {"status": "completed", "result": task.get()}
    return {"status": "processing"}

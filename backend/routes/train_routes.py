from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path
import uuid
import json
from tasks.train import train_and_infer_from_json
from celery.result import AsyncResult
from celery_worker import celery_app

router = APIRouter()

# 학습 요청 모델 (experiment_name 추가)
class TrainRequest(BaseModel):
    experiment_name: str
    structure_id: str  # 검증된 구조의 ID
    input_text: str
    max_length: int = Field(default=16, ge=1, le=1024)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_k: int = Field(default=40, ge=1, le=1000)
    dataset_name: str = "tiny_shakespeare"
    dataset_config: str = "default"

@router.post("/train")
async def train_model(request: TrainRequest):
    try:
        # 저장된 구조 불러오기
        STRUCTURE_DIR = Path("temp_structures")
        structure_path = STRUCTURE_DIR / f"{request.structure_id}.json"
        if not structure_path.exists():
            raise HTTPException(status_code=404, detail="Structure not found")
        with open(structure_path, "r") as f:
            layer_json = json.load(f)
        config = layer_json[0]
        layer_dicts = layer_json[1:]
        epochs = config.get('epochs', 5)
        dtype = config.get('dtype', 'fp32')
        # 학습 태스크 시작 (experiment_name을 첫 번째 인자로 전달)
        train_task = train_and_infer_from_json.apply_async(args=[
            request.experiment_name,
            layer_dicts,
            request.input_text,
            request.max_length,
            request.temperature,
            request.top_k,
            request.dataset_name,
            request.dataset_config,
            dtype,
            epochs
        ])
        return {
            "status": "success",
            "task_id": train_task.id,
            "message": "Training started successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{task_id}")
def get_status(task_id: str):
    try:
        task = AsyncResult(task_id, app=celery_app)
        return {
            "task_id": task_id,
            "status": task.state,
            "result": task.result if task.ready() else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

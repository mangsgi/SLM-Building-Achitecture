# routes/train_routes.py
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import json
import os
import socket
from urllib.parse import urlparse

import mlflow
from mlflow.tracking import MlflowClient

router = APIRouter()

# ===== 유틸 =====
def _can_connect_http(uri: str, timeout: float = 1.5) -> bool:
    try:
        u = urlparse(uri)
        if u.scheme not in ("http", "https") or not u.hostname:
            return True
        port = u.port or (443 if u.scheme == "https" else 80)
        with socket.create_connection((u.hostname, port), timeout=timeout):
            return True
    except Exception:
        return False

# ===== 모델 구성요소 =====
class ModelConfig(BaseModel):
    model: str
    epochs: int
    batch_size: int
    vocab_size: int
    context_length: int
    stride: int
    emb_dim: int
    n_heads: int
    n_blocks: int
    qkv_bias: Optional[bool] = None     # GPT-2
    drop_rate: Optional[float] = None   # GPT-2
    hidden_dim: Optional[int] = None    # Llama2, Llama3
    rope_base: Optional[float] = None   # Llama3 GQA
    rope_config: Optional[dict] = None  # Llama3 GQA
    n_kv_groups: Optional[int] = None   # Llama3 GQA
    qk_norm: Optional[bool] = None      # Qwen3 GQA
    qk_norm_eps: Optional[float] = None # Qwen3 GQA
    head_dim: Optional[int] = None      # Qwen3 GQA
    dtype: str

# ===== 노드 구성요소 =====
class LayerData(BaseModel):
    id: str
    label: Optional[str] = None
    inDim: Optional[int] = None
    outDim: Optional[int] = None
    vocabSize: Optional[int] = None         # for Tokenizer
    embDim: Optional[int] = None
    ctxLength: Optional[int] = None
    dropoutRate: Optional[float] = None     # for Dropout, Attention
    source: Optional[str] = None            # for Residual
    normType: Optional[str] = None          # for Normalization
    bias: Optional[bool] = None             # for FeedForward, Linear
    hiddenDim: Optional[int] = None         # for FeedForward
    feedForwardType: Optional[str] = None   # for FeedForward
    actFunc: Optional[str] = None           # for FeedForward
    numHeads: Optional[int] = None          # for Attention
    numOfBlocks: Optional[int] = None       # for TransformerBlock
    weightTying: Optional[bool] = None      # for Linear
    qkvBias: Optional[bool] = None          # for Attention
    projBias: Optional[bool] = None         # for Attention (미구현)
    isRoPE: Optional[bool] = None           # for RoPE in Attention
    ropeBase: Optional[float] = None        # for RoPE in Attention
    ropeConfig: Optional[dict] = None       # for RoPE in Attention
    numKvGroups: Optional[int] = None       # for Grouped Query Attention (미구현)
    qkNorm: Optional[bool] = None           # for Attention
    qkNormEps: Optional[float] = None       # for Attention
    headDim: Optional[int] = None           # for Attention

class LayerNode(BaseModel):
    type: str
    data: LayerData
    children: Optional[List['LayerNode']] = None

LayerNode.update_forward_refs()

class CompleteModelRequest(BaseModel):
    config: ModelConfig
    model: List[LayerNode]
    dataset: str
    modelName: str
    dataset_config: str = "default"

# ===== 학습 엔드포인트 =====
@router.post("/train-complete-model")
async def train_complete_model(request: CompleteModelRequest, http_request: Request):
    """
    모델 구조 검증 → 모델 생성(테스트) → 구조 저장(modelName.json) →
    MLflow 실험/런 생성(라우터) → Celery 학습 태스크 시작 → SSE로 진행/완료 알림
    """
    try:
        # ---- 지연 import (윈도우 --reload 안정화) ----
        from tasks.structure import validate_model_structure
        from ml.models.factory import build_model_from_json
        from tasks.train import train_and_infer_from_json

        # ---------- A) 입력 정리 ----------
        layer_dicts = [layer.dict() for layer in request.model]
        complete_structure = [request.config.dict()] + layer_dicts

        # ---------- B) 모델 구조 검증 (Celery) ----------
        print("1단계: 모델 구조 검증 중...")
        validation_async = validate_model_structure.apply_async(args=[layer_dicts])
        validation = validation_async.get(timeout=30)
        if validation.get("status") != "success":
            raise HTTPException(status_code=400, detail=f"모델 구조 검증 실패: {validation.get('message')}")

        # ---------- C) 모델 생성 테스트 ----------
        print("2단계: 모델 생성 테스트 중...")
        try:
            model = build_model_from_json(layer_dicts, dtype=request.config.dtype)
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"모델 생성 실패: {str(e)}")

        # ---------- D) 구조 파일 저장 ----------
        print("3단계: 모델 구조 저장 중...")
        STRUCTURE_DIR = Path("temp_structures")
        STRUCTURE_DIR.mkdir(exist_ok=True)
        structure_path = STRUCTURE_DIR / f"{request.modelName}.json"
        with open(structure_path, "w", encoding="utf-8") as f:
            json.dump(complete_structure, f, ensure_ascii=False)

        # ---------- E) MLflow: 실험/런 생성 (라우터에서 선생성) ----------
        print("4단계: 학습 시작...")
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000").rstrip("/")
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
        mlflow.set_tracking_uri(tracking_uri)

        if not _can_connect_http(tracking_uri):
            raise HTTPException(
                status_code=502,
                detail=f"MLflow 서버({tracking_uri})에 연결할 수 없습니다. "
                       f"MLFLOW_TRACKING_URI를 확인하거나 서버를 기동하세요."
            )

        experiment_name = request.modelName
        exp = mlflow.get_experiment_by_name(experiment_name)
        exp_id = exp.experiment_id if exp else mlflow.create_experiment(experiment_name)

        client = MlflowClient()
        run = client.create_run(
            experiment_id=exp_id,
            tags={
                "model_name": request.modelName,
                "structure_file": structure_path.name,
            }
        )
        run_id = run.info.run_id
        mlflow_url = f"{tracking_uri}/#/experiments/{exp_id}/runs/{run_id}"
        
        # 브라우저 접속용 URL 생성
        browser_mlflow_url = mlflow_url.replace("://mlflow:", "://localhost:")

        # ---------- F) 학습 태스크 시작 (run_id 포함) ----------
        payload = {
            "config": request.config.dict(exclude_none=True),
            "model": layer_dicts,
            "dataset": request.dataset,
            "modelName": request.modelName,
            "dataset_config": request.dataset_config,
            "experiment_name": experiment_name,
            "run_id": run_id,
        }
        train_task = train_and_infer_from_json.apply_async(args=[payload])

        # ---------- G) 절대 URL 제공 ----------
        base = str(http_request.base_url).rstrip("/")
        sse_path = f"/api/v1/events/{train_task.id}"
        sse_url_abs = f"{base}{sse_path}"
        stop_url_abs = f"{base}/api/v1/stop-training"

        return {
            "status": "success",
            "task_id": train_task.id,
            "sse_url": sse_path,         # 상대 경로
            "sse_url_abs": sse_url_abs,  # 절대 경로
            "stop_url_abs": stop_url_abs,
            "structure_id": request.modelName,
            "model_name": request.modelName,
            "experiment_name": experiment_name,
            "mlflow_url": browser_mlflow_url,
            "model_info": {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "config": request.config.dict(exclude_none=True),
                "dataset": request.dataset
            },
            "training_config": {
                "dataset": request.dataset,
                "dataset_config": request.dataset_config,
                "epochs": request.config.epochs
            },
            "message": "모델 검증/생성/구조저장/MLflow 런 생성 후 학습을 시작했습니다. 진행상황은 SSE로 전송됩니다."
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"통합 처리 중 오류 발생: {str(e)}")

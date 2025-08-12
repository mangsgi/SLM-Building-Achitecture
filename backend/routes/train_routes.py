from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from pathlib import Path
import json
from tasks.train import train_and_infer_from_json
from tasks.structure import validate_model_structure
from celery.result import AsyncResult
from celery_worker import celery_app
from ml.models.factory import build_model_from_json
import mlflow
from mlflow.tracking import MlflowClient

router = APIRouter()

# ===== 요청 모델 =====
class ModelConfig(BaseModel):
    model: str
    epochs: int
    batch_size: int
    vocab_size: int
    context_length: int
    emb_dim: int
    n_heads: int
    n_blocks: int
    drop_rate: float
    qkv_bias: bool
    dtype: str

class LayerData(BaseModel):
    id: str
    label: str = None
    inDim: int = None
    outDim: int = None
    vocabSize: int = None
    embDim: int = None
    ctxLength: int = None
    dropoutRate: float = None
    numOfFactor: int = None
    source: str = None
    numHeads: int = None
    qkvBias: bool = None
    numOfBlocks: int = None
    numKvGroups: int = None

class LayerNode(BaseModel):
    type: str
    data: LayerData
    children: List['LayerNode'] = None

LayerNode.update_forward_refs()

class CompleteModelRequest(BaseModel):
    config: ModelConfig
    model: List[LayerNode]
    dataset: str
    modelName: str
    dataset_config: str = "default"

# ===== 학습 엔드포인트 =====
@router.post("/train-complete-model")
async def train_complete_model(request: CompleteModelRequest):
    """
    모델 구조 검증 → 모델 생성(테스트) → 구조 저장(modelName.json) → 학습 시작
    """
    try:
        # 데이터 준비
        layer_dicts = [layer.dict() for layer in request.model]
        complete_structure = [request.config.dict()] + layer_dicts

        # 1단계: 모델 구조 검증 (Celery task)
        print("1단계: 모델 구조 검증 중...")
        validation_result = validate_model_structure.apply_async(args=[layer_dicts])
        validation_response = validation_result.get(timeout=30)
        if validation_response["status"] != "success":
            raise HTTPException(status_code=400, detail=f"모델 구조 검증 실패: {validation_response['message']}")
        print("✅ 모델 구조 검증 완료")

        # 2단계: 모델 생성 테스트 (학습과 동일하게 layer_dicts만 전달)
        print("2단계: 모델 생성 테스트 중...")
        try:
            model = build_model_from_json(layer_dicts, dtype=request.config.dtype)
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"✅ 모델 생성 완료 - 총 파라미터: {total_params:,}, 학습 가능 파라미터: {trainable_params:,}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"모델 생성 실패: {str(e)}")

        # 3단계: 구조 저장 (modelName.json) - config + layers 저장 유지
        print("3단계: 모델 구조 저장 중...")
        STRUCTURE_DIR = Path("temp_structures")
        STRUCTURE_DIR.mkdir(exist_ok=True)
        structure_path = STRUCTURE_DIR / f"{request.modelName}.json"
        if structure_path.exists():
            print(f"⚠ 경고: {structure_path.name} 파일이 이미 존재하여 덮어씁니다.")
        with open(structure_path, "w", encoding="utf-8") as f:
            json.dump(complete_structure, f, ensure_ascii=False)
        print(f"✅ 모델 구조 저장 완료 - 파일명: {structure_path.name}")

        # 4단계: MLflow 실험/런 생성 + 학습 태스크 시작
        print("4단계: 학습 시작...")
        experiment_name = request.modelName
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
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
        tracking_uri = mlflow.get_tracking_uri().rstrip("/")
        mlflow_url = f"{tracking_uri}/#/experiments/{exp_id}/runs/{run_id}"

        payload = {
            "config": request.config.dict(),
            "model": layer_dicts,              # 학습에도 layer_dicts만 전달
            "dataset": request.dataset,
            "modelName": request.modelName,
            "dataset_config": request.dataset_config,
            "experiment_name": experiment_name,
            "run_id": run_id,
        }

        train_task = train_and_infer_from_json.apply_async(args=[payload])

        return {
            "status": "success",
            "task_id": train_task.id,
            "structure_id": request.modelName,
            "model_name": request.modelName,
            "experiment_name": experiment_name,
            "mlflow_url": mlflow_url,
            "model_info": {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "config": request.config.dict(),
                "dataset": request.dataset
            },
            "training_config": {
                "dataset": request.dataset,
                "dataset_config": request.dataset_config,
                "epochs": request.config.epochs
            },
            "message": "모델 검증, 생성, 학습이 모두 성공적으로 시작되었습니다."
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"통합 처리 중 오류 발생: {str(e)}")

# ===== 학습 상태 조회 =====
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

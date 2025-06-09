from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from celery.result import AsyncResult
from tasks import train_and_infer_from_json
from celery_worker import celery_app

router = APIRouter()

# 기본 레이어 데이터 모델
class BaseLayerData(BaseModel):
    id: str

# 토큰 임베딩 데이터
class TokenEmbeddingData(BaseLayerData):
    vocab_size: int
    d_model: int

# 포지셔널 임베딩 데이터
class PositionalEmbeddingData(BaseLayerData):
    d_model: int
    max_len: int
    mode: str = "learned"

# 어텐션 데이터
class AttentionData(BaseLayerData):
    d_in: int
    d_out: int
    num_heads: int
    context_length: int
    attn_type: str = "default"  # "default", "flash", "gqa"
    num_kv_groups: Optional[int] = None  # GQA를 위한 파라미터
    dropout: float = 0.0
    qkv_bias: bool = False
    rope_base: int = 10000  # RoPE 기본값
    rope_config: Optional[Dict[str, Any]] = None  # RoPE 추가 설정
    dtype: Optional[str] = None  # 데이터 타입 (예: "float32", "float16")

    @validator('num_kv_groups')
    def validate_num_kv_groups(cls, v, values):
        if v is not None:
            num_heads = values.get('num_heads')
            if num_heads % v != 0:
                raise ValueError("num_heads must be divisible by num_kv_groups")
        return v

    @validator('attn_type')
    def validate_attn_type(cls, v, values):
        valid_types = {"default", "flash", "gqa"}
        if v not in valid_types:
            raise ValueError(f"Invalid attention type: {v}. Must be one of {valid_types}")
        
        # GQA 타입일 때는 num_kv_groups가 필수
        if v == "gqa" and values.get('num_kv_groups') is None:
            raise ValueError("num_kv_groups is required for GQA attention type")
        
        return v

# 피드포워드 데이터
class FeedForwardData(BaseLayerData):
    emb_dim: int
    dff_ratio: float = 4.0
    activation: str = "relu"

# 정규화 데이터
class NormalizationData(BaseLayerData):
    d_model: int

# 잔차 연결 데이터
class ResidualData(BaseLayerData):
    source: str

# 동적 블록 데이터
class DynamicBlockData(BaseLayerData):
    numLayers: int = 1

# 레이어 노드 모델
class LayerNode(BaseModel):
    type: str
    data: Dict[str, Any]  # 모든 데이터 타입을 허용
    children: Optional[List['LayerNode']] = None

    @validator('type')
    def validate_type(cls, v):
        valid_types = {
            "tokenEmbedding", "positionalEmbedding", "normalization",
            "attention", "feedForward", "residual", "dynamicBlock",
            "linear", "dropout"
        }
        if v not in valid_types:
            raise ValueError(f"Invalid layer type: {v}. Must be one of {valid_types}")
        return v

    @validator('data')
    def validate_data(cls, v, values):
        layer_type = values.get('type')
        if not layer_type:
            return v

        required_fields = {
            "tokenEmbedding": {"vocab_size", "d_model"},
            "positionalEmbedding": {"d_model", "max_len"},
            "normalization": {"d_model"},
            "attention": {"d_in", "d_out", "num_heads", "context_length"},
            "feedForward": {"emb_dim"},
            "residual": {"source"},
            "dynamicBlock": set(),  # 동적 블록은 필수 필드가 없음
            "linear": {"inDim", "outDim"},
            "dropout": set()  # 드롭아웃은 필수 필드가 없음
        }

        if layer_type in required_fields:
            missing_fields = required_fields[layer_type] - set(v.keys())
            if missing_fields:
                raise ValueError(f"Missing required fields for {layer_type}: {missing_fields}")

        return v

    @validator('children')
    def validate_children(cls, v, values):
        if values.get('type') == 'dynamicBlock' and not v:
            raise ValueError("dynamicBlock must have children")
        return v

LayerNode.update_forward_refs()

# 학습 요청 모델
class TrainRequest(BaseModel):
    exp_name: str
    layer_json: List[LayerNode]
    input_text: str
    max_length: int = Field(default=16, ge=1, le=1024)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_k: int = Field(default=40, ge=1, le=1000)
    dataset_name: str = "tiny_shakespeare"
    dataset_config: str = "default"

    @validator('layer_json')
    def validate_layer_json(cls, v):
        if not v:
            raise ValueError("layer_json cannot be empty")
        
        # 첫 번째 레이어는 반드시 tokenEmbedding이어야 함
        if v[0].type != "tokenEmbedding":
            raise ValueError("First layer must be tokenEmbedding")
            
        # 두 번째 레이어는 반드시 positionalEmbedding이어야 함
        if len(v) > 1 and v[1].type != "positionalEmbedding":
            raise ValueError("Second layer must be positionalEmbedding")
            
        return v

@router.post("/train")
async def train_model(request: TrainRequest):
    try:
        # JSON을 딕셔너리로 변환
        layer_dicts = [layer.dict() for layer in request.layer_json]
        
        # 학습 태스크 시작
        train_task = train_and_infer_from_json.apply_async(args=[
            request.exp_name,
            layer_dicts,
            request.input_text,
            request.max_length,
            request.temperature,
            request.top_k,
            request.dataset_name,
            request.dataset_config
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

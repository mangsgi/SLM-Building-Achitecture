from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, validator
from typing import List, Dict, Any
import uuid
import json
from pathlib import Path
from tasks.structure import validate_model_structure

router = APIRouter()

STRUCTURE_DIR = Path("temp_structures")
STRUCTURE_DIR.mkdir(exist_ok=True)

# 구조 검증에 필요한 Pydantic 모델 및 검증 로직 추가
class BaseLayerData(BaseModel):
    id: str
    label: str = None
    inDim: int = None
    outDim: int = None

class TokenEmbeddingData(BaseLayerData):
    vocabSize: int
    embDim: int

class PositionalEmbeddingData(BaseLayerData):
    embDim: int
    ctxLength: int

class DropoutData(BaseLayerData):
    dropoutRate: float = 0.1

class TransformerBlockData(BaseLayerData):
    numOfBlocks: int = 1

class NormalizationData(BaseLayerData):
    pass

class FeedForwardData(BaseLayerData):
    numOfFactor: int = 4

class ResidualData(BaseLayerData):
    source: str

class LinearData(BaseLayerData):
    pass

class AttentionData(BaseLayerData):
    num_heads: int
    ctxlength: int
    attn_type: str = "default"
    num_kv_groups: int = None
    dropout: float = 0.0
    qkv_bias: bool = False
    rope_base: int = 10000
    rope_config: dict = None
    dtype: str = None

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
        if v == "gqa" and values.get('num_kv_groups') is None:
            raise ValueError("num_kv_groups is required for GQA attention type")
        return v

class LayerNode(BaseModel):
    type: str
    data: Dict[str, Any]
    children: List['LayerNode'] = None

    @validator('type')
    def validate_type(cls, v):
        valid_types = {
            "tokenEmbedding", "positionalEmbedding", "normalization",
            "attention", "feedForward", "residual", "transformerBlock",
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
            "tokenEmbedding": {"vocabSize", "embDim", "id"},
            "positionalEmbedding": {"embDim", "ctxLength", "id"},
            "normalization": {"id"},
            "attention": {"inDim", "outDim", "num_heads", "ctxLength", "id"},
            "feedForward": {"numOfFactor", "id"},
            "residual": {"source", "id"},
            "transformerBlock": {"numOfBlocks", "id"},
            "linear": {"inDim", "outDim", "id"},
            "dropout": {"dropoutRate", "id"}
        }
        if layer_type in required_fields:
            missing_fields = required_fields[layer_type] - set(v.keys())
            if missing_fields:
                raise ValueError(f"Missing required fields for {layer_type}: {missing_fields}")
        return v

    @validator('children')
    def validate_children(cls, v, values):
        if values.get('type') == 'transformerBlock' and not v:
            raise ValueError("transformerBlock must have children")
        return v

LayerNode.update_forward_refs()

class StructureValidationRequest(BaseModel):
    layer_json: List[Dict[str, Any]]

    @validator('layer_json')
    def validate_layer_json(cls, v):
        if not v or not isinstance(v, list):
            raise ValueError("layer_json must be a non-empty list")
        config_obj = v[0]
        if 'type' in config_obj:
            raise ValueError("First object in layer_json must be config (no 'type' field)")
        for node in v[1:]:
            LayerNode(**node)
        if v[1]['type'] != "tokenEmbedding":
            raise ValueError("First layer must be tokenEmbedding")
        if len(v) > 2 and v[2]['type'] != "positionalEmbedding":
            raise ValueError("Second layer must be positionalEmbedding")
        return v

@router.post("/validate-structure")
async def validate_structure(request: StructureValidationRequest):
    try:
        layer_dicts = request.layer_json[1:]
        result = validate_model_structure.apply_async(args=[layer_dicts])
        celery_result = result.get(timeout=30)

        if celery_result["status"] != "success":
            raise HTTPException(status_code=500, detail=celery_result["message"])

        structure_id = str(uuid.uuid4())
        structure_path = STRUCTURE_DIR / f"{structure_id}.json"
        with open(structure_path, "w") as f:
            json.dump(request.layer_json, f)

        return {
            "status": "success",
            "structure_id": structure_id,
            "structure": celery_result["structure"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"구조 검증 또는 모델 생성 오류: {e}") 
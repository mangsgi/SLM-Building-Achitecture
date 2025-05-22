import torch
import torch.nn as nn
from typing import Dict, Any, Union, List

from .components.token_embedding import TokenEmbedding
from .components.positional_embedding import (
    LearnedPositionalEmbedding,
    SinusoidalPositionalEmbedding,
    RelativePositionalEmbedding,
    RotaryPositionalEmbedding,
)
from .components.attention import MultiHeadAttentionCombinedQKV, MHAPyTorchScaledDotProduct, GroupedQueryAttention
from .components.normalization import LayerNorm, RMSNorm
from .components.ffn import CustomFFN
from .components.residual import ResidualConnection
from .components.dynamic_block import DynamicBlock


# ===== Factory Classes =====
class LayerFactory:
    """메인 팩토리 클래스: 모든 레이어 생성을 담당"""
    @staticmethod
    def create_layer(node: Dict[str, Any]) -> nn.Module:
        layer_type = node["type"]
        data = node["data"]
        
        factory_map = {
            "tokenEmbedding": TokenEmbeddingFactory,
            "positionalEmbedding": PositionalEmbeddingFactory,
            "normalization": NormalizationFactory,
            "attention": AttentionFactory,
            "feedForward": FeedForwardFactory,
            "residual": ResidualFactory,
            "dynamicBlock": DynamicBlockFactory,
            "linear": LinearFactory,
            "dropout": DropoutFactory
        }
        
        factory = factory_map.get(layer_type)
        if factory is None:
            raise ValueError(f"Unknown layer type: {layer_type}")
            
        return factory.create(data if layer_type != "dynamicBlock" else node)


class TokenEmbeddingFactory:
    """토큰 임베딩 레이어 생성"""
    @staticmethod
    def create(data: Dict[str, Any]) -> TokenEmbedding:
        return TokenEmbedding(
            vocab_size=data["vocab_size"],
            d_model=data["d_model"]
        )


class PositionalEmbeddingFactory:
    """포지셔널 임베딩 레이어 생성"""
    @staticmethod
    def create(data: Dict[str, Any]) -> Union[
        LearnedPositionalEmbedding,
        SinusoidalPositionalEmbedding,
        RelativePositionalEmbedding,
        RotaryPositionalEmbedding
    ]:
        mode = data.get("mode", "learned")
        if mode == "learned":
            return LearnedPositionalEmbedding(data["max_len"], data["d_model"])
        elif mode == "sinusoidal":
            return SinusoidalPositionalEmbedding(data["max_len"], data["d_model"])
        elif mode == "relative":
            return RelativePositionalEmbedding(data["max_len"], data["d_model"])
        elif mode == "rotary":
            return RotaryPositionalEmbedding(data["d_model"], data["max_len"])
        else:
            raise ValueError(f"Unknown positional embedding mode: {mode}")


class NormalizationFactory:
    """정규화 레이어 생성"""
    @staticmethod
    def create(data: Dict[str, Any]) -> LayerNorm:
        return LayerNorm(data["d_model"])


class AttentionFactory:
    """어텐션 레이어 생성"""
    @staticmethod
    def create(data: Dict[str, Any]) -> Union[MultiHeadAttentionCombinedQKV, MHAPyTorchScaledDotProduct, GroupedQueryAttention]:
        attn_type = data.get("attn_type", "default")
        if attn_type == "default":
            return MultiHeadAttentionCombinedQKV(
                d_in=data["d_in"],
                d_out=data["d_out"],
                num_heads=data["num_heads"],
                context_length=data["context_length"],
                dropout=data.get("dropout", 0.0),
                qkv_bias=data.get("qkv_bias", False)
            )
        elif attn_type == "flash":
            return MHAPyTorchScaledDotProduct(
                d_in=data["d_in"],
                d_out=data["d_out"],
                num_heads=data["num_heads"],
                context_length=data["context_length"],
                dropout=data.get("dropout", 0.0),
                qkv_bias=data.get("qkv_bias", False)
            )
        elif attn_type == "gqa":
            return GroupedQueryAttention(
                d_in=data["d_in"],
                d_out=data["d_out"],
                context_length=data["context_length"],
                num_heads=data["num_heads"],
                num_kv_groups=data["num_kv_groups"],
                rope_base=data.get("rope_base", 10000),
                rope_config=data.get("rope_config"),
                dtype=data.get("dtype")
            )
        else:
            raise ValueError(f"Unknown attention type: {attn_type}")


class FeedForwardFactory:
    """피드포워드 레이어 생성"""
    @staticmethod
    def create(data: Dict[str, Any]) -> CustomFFN:
        return CustomFFN(
            emb_dim=data["emb_dim"],
            dff_ratio=data.get("dff_ratio", 4.0),
            activation=data.get("activation", "relu"),
            gated=data.get("gated", False),
            dtype=torch.float32
        )


class ResidualFactory:
    """잔차 연결 레이어 생성"""
    @staticmethod
    def create(data: Dict[str, Any]) -> ResidualConnection:
        if "source" not in data:
            raise ValueError(f"Residual layer '{data.get('id', 'unknown')}' must have a 'source' field")
        return ResidualConnection(data["source"])


class DynamicBlockFactory:
    """동적 블록 레이어 생성"""
    @staticmethod
    def create(node: Dict[str, Any]) -> DynamicBlock:
        children = [LayerFactory.create_layer(child) for child in node.get("children", [])]
        num_layers = node["data"].get("numLayers", 1)
        block_id = node["data"].get("id")
        return DynamicBlock(*children, num_layers=num_layers, block_id=block_id)


class LinearFactory:
    """선형 레이어 생성"""
    @staticmethod
    def create(data: Dict[str, Any]) -> nn.Linear:
        return nn.Linear(data["inDim"], data["outDim"])


class DropoutFactory:
    """드롭아웃 레이어 생성"""
    @staticmethod
    def create(data: Dict[str, Any]) -> nn.Dropout:
        return nn.Dropout(data.get("dropout_rate", 0.1))


# ===== Model Classes =====
class CustomSequential(nn.Module):
    """커스텀 시퀀셜 모델: 레이어 간 연결과 캐싱을 처리"""
    def __init__(self, layer_list: List[nn.Module], id_to_module_map: Dict[str, nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layer_list)
        self.id_to_module = id_to_module_map

    def forward(self, x):
        cache = {}
        prev_out = None
        for i, layer in enumerate(self.layers):
            # 1. TokenEmbedding → PositionalEmbedding 연속 시 자동 합산
            if (
                i > 0
                and isinstance(layer, (LearnedPositionalEmbedding, SinusoidalPositionalEmbedding, RelativePositionalEmbedding, RotaryPositionalEmbedding))
                and isinstance(self.layers[i-1], TokenEmbedding)
            ):
                pos_out = layer(x)
                x = prev_out + pos_out
                
            # 2. ResidualConnection 처리
            elif isinstance(layer, ResidualConnection):
                source_id = layer.source_id
                if source_id in cache:
                    x = x + cache[source_id]
                else:
                    # dynamicBlock 내부의 레이어 ID도 확인
                    block_source_id = f"{layer.source_id}_layer_{i}" if hasattr(layer, 'block_id') else layer.source_id
                    if block_source_id in cache:
                        x = x + cache[block_source_id]
                    else:
                        # 이전 출력을 사용
                        x = x + prev_out
            else:
                x = layer(x)
                
            prev_out = x
            if hasattr(layer, "layer_id"):
                cache[layer.layer_id] = x
        return x


# ===== Public API =====
def build_model_from_json(json_list: List[Dict[str, Any]]) -> CustomSequential:
    """JSON으로부터 모델을 생성하는 메인 함수"""
    id_map = {}
    id_to_module = {}

    for node in json_list:
        id_map[node["data"]["id"]] = node

    layers = []
    for node in json_list:
        layer = LayerFactory.create_layer(node)
        if "id" in node["data"]:
            layer.layer_id = node["data"]["id"]
            id_to_module[layer.layer_id] = layer
        layers.append(layer)

    return CustomSequential(layers, id_to_module)


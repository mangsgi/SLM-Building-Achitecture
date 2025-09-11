import torch
import torch.nn as nn
from typing import Dict, Any, Union, List
from .utils import str_to_torch_dtype

from .components.token_embedding import TokenEmbedding
from .components.positional_embedding import (
    LearnedPositionalEmbedding,
    SinusoidalPositionalEmbedding,
    RelativePositionalEmbedding,
    RotaryPositionalEmbedding,
)
from .components.attention import (
    MultiHeadAttentionCombinedQKV,
    MHAPyTorchScaledDotProduct,
    GroupedQueryAttention,
)
from .components.normalization import LayerNorm, RMSNorm
from .components.ffn import CustomFFN
from .components.residual import ResidualConnection
from .components.transformer_block import TrasnformerBlock


# ===== Factory Classes =====
class LayerFactory:
    """메인 팩토리 클래스: 모든 레이어 생성을 담당"""
    @staticmethod
    def create_layer(node: Dict[str, Any], dtype=torch.float32) -> nn.Module:
        layer_type = node["type"]
        data = node["data"].copy()
        data["dtype"] = dtype  # dtype을 data에 추가

        factory_map = {
            "tokenEmbedding": TokenEmbeddingFactory,
            "positionalEmbedding": PositionalEmbeddingFactory,
            "normalization": NormalizationFactory,
            "attention": AttentionFactory,
            "mhAttention": AttentionFactory,
            "flashAttention": AttentionFactory,
            "gqaAttention": AttentionFactory,
            "feedForward": FeedForwardFactory,
            "residual": ResidualFactory,
            "transformerBlock": TransformerBlockFactory,
            "linear": LinearFactory,
            "dropout": DropoutFactory,
        }

        factory = factory_map.get(layer_type)
        if factory is None:
            raise ValueError(f"Unknown layer type: {layer_type}")

        # AttentionFactory의 경우 layer_type을 추가로 전달
        if factory == AttentionFactory:
            return factory.create(
                data if layer_type != "transformerBlock" else node,
                dtype=dtype,
                layer_type=layer_type,
            )
        else:
            return factory.create(
                data if layer_type != "transformerBlock" else node, dtype=dtype
            )


class TokenEmbeddingFactory:
    """토큰 임베딩 레이어 생성"""
    @staticmethod
    def create(data: Dict[str, Any], dtype=torch.float32) -> TokenEmbedding:
        return TokenEmbedding(
            vocab_size=data["vocabSize"],
            d_model=data["embDim"],
            dtype=dtype,  # dtype 추가
        )


class PositionalEmbeddingFactory:
    """포지셔널 임베딩 레이어 생성"""
    @staticmethod
    def create(data: Dict[str, Any], dtype=torch.float32):
        mode = data.get("mode", "Learned Positional Embedding")
        common_args = {
            "ctx_length": data["ctxLength"],
            "emb_dim": data["embDim"],
            "dtype": dtype,  # dtype 추가
        }

        if mode == "Learned Positional Embedding":
            return LearnedPositionalEmbedding(**common_args)
        elif mode == "Sinusoidal Positional Embedding":
            return SinusoidalPositionalEmbedding(**common_args)
        elif mode == "Relative Positional Embedding":
            return RelativePositionalEmbedding(**common_args)
        elif mode == "Rotary Positional Embedding":
            return RotaryPositionalEmbedding(
                data["embDim"], data["ctxLength"], dtype=dtype
            )
        else:
            raise ValueError(f"Unknown positional embedding mode: {mode}")


class NormalizationFactory:
    """정규화 레이어 생성"""
    @staticmethod
    def create(data: Dict[str, Any], dtype=torch.float32) -> LayerNorm:
        d_model = data.get("embDim") or data.get("inDim") or data.get("outDim")
        return LayerNorm(d_model, dtype=dtype)  # dtype 추가


class AttentionFactory:
    """어텐션 레이어 생성 (새로운 타입별 처리)"""
    @staticmethod
    def create(data: Dict[str, Any], dtype=torch.float32, layer_type: str = None):
        # 공통 인자 설정
        common_args = {
            "d_in": data.get("inDim") or data.get("embDim"),
            "d_out": data.get("outDim") or data.get("embDim"),
            "context_length": data["ctxLength"],
            "dtype": dtype,
        }

        # 타입별 처리
        if layer_type == "mhAttention" or (
            layer_type is None and "numHeads" in data
        ):
            # Scaled Dot-Product Attention
            common_args.update(
                {
                    "num_heads": data["numHeads"],
                    "dropout": data.get("dropoutRate", 0.0),
                    "qkv_bias": data.get("qkvBias", False),
                }
            )
            return MHAPyTorchScaledDotProduct(**common_args)

        elif layer_type == "flashAttention":
            # Flash Attention (SDPA 백엔드 사용)
            common_args.update(
                {
                    "num_heads": data["numHeads"],
                    "dropout": data.get("dropoutRate", 0.0),
                    "qkv_bias": data.get("qkvBias", False),
                }
            )
            return MHAPyTorchScaledDotProduct(**common_args)

        elif layer_type == "gqaAttention":
            # Grouped Query Attention (+ RoPE)
            common_args.update(
                {
                    "num_heads": data["numHeads"],
                    "dropout": data.get("dropoutRate", 0.0),
                    "qkv_bias": data.get("qkvBias", False),  # 일단 받되…
                }
            )
            # GQA는 qkv_bias 인자를 사용하지 않으므로 제거
            common_args.pop("qkv_bias", None)

            # RoPE는 head_dim= d_out/num_heads가 짝수여야 함
            head_dim = (common_args["d_out"] // common_args["num_heads"])
            if head_dim % 2 != 0:
                raise ValueError(
                    f"head_dim({head_dim}) must be even for RoPE (GQA)."
                )

            return GroupedQueryAttention(
                **common_args,
                num_kv_groups=data["numKvGroups"],
                rope_base=data.get("ropeBase", 10000),
                rope_config=data.get("ropeConfig"),
            )

        else:  # 기존 attention 타입 (하위 호환성)
            attn_type = data.get("attn_type", "default")
            common_args.update(
                {
                    "num_heads": data["num_heads"],
                    "dropout": data.get("dropout", 0.0),
                    "qkv_bias": data.get("qkv_bias", False),
                }
            )

            if attn_type == "default":
                return MultiHeadAttentionCombinedQKV(**common_args)
            elif attn_type == "flash":
                return MHAPyTorchScaledDotProduct(**common_args)
            elif attn_type == "gqa":
                # 레거시 키 이름을 쓰는 경우도 동일 처리
                head_dim = (common_args["d_out"] // common_args["num_heads"])
                if head_dim % 2 != 0:
                    raise ValueError(
                        f"head_dim({head_dim}) must be even for RoPE (GQA)."
                    )
                return GroupedQueryAttention(
                    **{k: v for k, v in common_args.items() if k != "qkv_bias"},
                    num_kv_groups=data["num_kv_groups"],
                    rope_base=data.get("rope_base", 10000),
                    rope_config=data.get("rope_config"),
                )
            else:
                raise ValueError(f"Unknown attention type: {attn_type}")


class FeedForwardFactory:
    """피드포워드 레이어 생성"""
    @staticmethod
    def create(data: Dict[str, Any], dtype=torch.float32) -> CustomFFN:
        return CustomFFN(
            emb_dim=data.get("outDim") or data.get("inDim"),
            hidden_dim=data.get("hiddenDim", 3072),
            activation=data.get("actFunc", "GELU"),
            is_gated=data.get("feedForwardType", "Standard") == "Gated",
            dtype=dtype,  # dtype 이미 있음
        )


class ResidualFactory:
    """잔차 연결 레이어 생성"""
    @staticmethod
    def create(data: Dict[str, Any], dtype=torch.float32) -> ResidualConnection:
        if "source" not in data:
            raise ValueError(
                f"Residual layer '{data.get('id', 'unknown')}' must have a 'source' field"
            )
        return ResidualConnection(data["source"])


class TransformerBlockFactory:
    """트랜스포머 블록 레이어 생성"""
    @staticmethod
    def create(node: Dict[str, Any], dtype=torch.float32) -> TrasnformerBlock:
        children = [
            LayerFactory.create_layer(child, dtype=dtype)
            for child in node.get("children", [])
        ]
        num_layers = node["data"].get("numOfBlocks", 1)
        block_id = node["data"].get("id")
        return TrasnformerBlock(*children, num_layers=num_layers, block_id=block_id)


class LinearFactory:
    """선형 레이어 생성"""
    @staticmethod
    def create(data: Dict[str, Any], dtype=torch.float32) -> nn.Linear:
        return nn.Linear(
            in_features=data["inDim"],
            out_features=data["outDim"],
            dtype=dtype,  # dtype 추가
        )


class DropoutFactory:
    """드롭아웃 레이어 생성"""
    @staticmethod
    def create(data: Dict[str, Any], dtype=torch.float32) -> nn.Dropout:
        return nn.Dropout(data.get("dropoutRate", 0.1))


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
            # 1) TokenEmbedding → PositionalEmbedding(learned/sinusoidal/relative) 자동 합산
            #    RotaryPositionalEmbedding 은 제외 (RoPE는 어텐션 내부에서 처리됨)
            if (
                i > 0
                and isinstance(self.layers[i - 1], TokenEmbedding)
                and isinstance(
                    layer,
                    (
                        LearnedPositionalEmbedding,
                        SinusoidalPositionalEmbedding,
                        RelativePositionalEmbedding,
                    ),
                )
            ):
                pos_out = layer(x)
                x = prev_out + pos_out

            # 2) ResidualConnection 처리
            elif isinstance(layer, ResidualConnection):
                source_id = layer.source_id
                if source_id in cache:
                    x = x + cache[source_id]
                else:
                    # TrasnformerBlock 내부의 레이어 ID도 확인
                    block_source_id = (
                        f"{layer.source_id}_layer_{i}"
                        if hasattr(layer, "block_id")
                        else layer.source_id
                    )
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
def build_model_from_json(
    json_list: List[Dict[str, Any]], dtype: str = "fp32"
) -> CustomSequential:
    """JSON으로부터 모델을 생성하는 메인 함수 (dtype 지원)"""
    id_map = {}
    id_to_module = {}

    torch_dtype = str_to_torch_dtype(dtype)

    # 첫 번째 객체(config)를 제외한 나머지 레이어들만 처리
    layer_nodes = [node for node in json_list if "type" in node]

    for node in layer_nodes:
        id_map[node["data"]["id"]] = node

    layers = []
    print("--- Starting model build loop ---")  # 디버깅용 print 추가
    for node in layer_nodes:
        node_id = node.get("data", {}).get("id", "N/A")
        print(f"--- Creating layer for node: {node_id} ---")  # 디버깅용 print 추가

        layer = LayerFactory.create_layer(node, dtype=torch_dtype)
        if "id" in node["data"]:
            layer.layer_id = node["data"]["id"]
            id_to_module[layer.layer_id] = layer
        layers.append(layer)

    print("--- Model build loop finished ---")  # 디버깅용 print 추가
    return CustomSequential(layers, id_to_module)

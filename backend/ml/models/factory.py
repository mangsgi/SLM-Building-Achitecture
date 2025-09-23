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
    MultiHeadAttentionUsingSDP,
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
            "gqAttention": AttentionFactory,
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
        
        if data.get("normType") == "Layer Normalization":
            return LayerNorm(d_model, dtype=dtype)  # dtype 추가
        elif data.get("normType") == "RMS Normalization":
            return RMSNorm(d_model, dtype=dtype)  # dtype 추가
        else:
            raise ValueError(f"Unknown normalization type: {data.get('normType')}")


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
                    "dropout": data.get("dropoutRate"),
                    "qkv_bias": data.get("qkvBias"),
                    "is_rope": data.get("isRoPE"),
                    "rope_base": data.get("ropeBase", 10000.0),
                }
            )
            return MultiHeadAttentionUsingSDP(**common_args)

        elif layer_type == "gqAttention":
            # Grouped Query Attention (+ RoPE)
            common_args.update(
                {
                    "num_heads": data["numHeads"],
                    "dropout": data.get("dropoutRate"),
                    "qkv_bias": data.get("qkvBias"),
                    "is_rope": data.get("isRoPE"),
                    "rope_base": data.get("ropeBase"),
                    "rope_config": data.get("ropeConfig"),
                    "num_kv_groups": data["numKvGroups"],
                }
            )
            
            return GroupedQueryAttention(**common_args)

        else:  
            # 기존 attention 타입 (하위 호환성)
            attn_type = data.get("attn_type", "default")
            common_args.update(
                {
                    "num_heads": data["numHeads"],
                    "dropout": data.get("dropoutRate"),
                    "qkv_bias": data.get("qkvBias"),
                    "is_rope": data.get("isRoPE"),
                    "rope_base": data.get("ropeBase"),
                    "rope_config": data.get("ropeConfig"),
                    "num_kv_groups": data["numKvGroups"],
                }
            )

            if attn_type == "default" or attn_type == "mha":
                return MultiHeadAttentionUsingSDP(**common_args)
            elif attn_type == "gqa":
                # 레거시 키 이름을 쓰는 경우도 동일 처리
                head_dim = (common_args["d_out"] // common_args["numHeads"])
                if head_dim % 2 != 0:
                    raise ValueError(
                        f"head_dim({head_dim}) must be even for RoPE (GQA)."
                    )
                return GroupedQueryAttention(
                    **{k: v for k, v in common_args.items() if k != "qkvBias"},
                    num_kv_groups=data["numKvGroups"],
                    rope_base=data.get("ropeBase"),
                    rope_config=data.get("ropeConfig"),
                )
            else:
                raise ValueError(f"Unknown attention type: {attn_type}")


class FeedForwardFactory:
    """피드포워드 레이어 생성"""
    @staticmethod
    def create(data: Dict[str, Any], dtype=torch.float32) -> CustomFFN:
        if data.get("actFunc") is None:
            raise ValueError(f"FeedForward layer '{data.get('id', 'unknown')}' must have an 'actFunc' field")
        if data.get("feedForwardType") is None:
            raise ValueError(f"FeedForward layer '{data.get('id', 'unknown')}' must have a 'feedForwardType' field")
        if data.get("bias") is None:
            raise ValueError(f"FeedForward layer '{data.get('id', 'unknown')}' must have a 'bias' field")
        if data.get("hiddenDim") is None:
            raise ValueError(f"FeedForward layer '{data.get('id', 'unknown')}' must have a 'hiddenDim' field")
        
        return CustomFFN(
            emb_dim=data.get("outDim") or data.get("inDim"),
            hidden_dim=data.get("hiddenDim"),
            activation=data.get("actFunc"),
            is_gated=data.get("feedForwardType") == "Gated",
            bias=data.get("bias"),
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
        if data.get("bias") is None:
            raise ValueError(f"Linear layer '{data.get('id', 'unknown')}' must have a 'bias' field")
        if data.get("weightTying") is None:
            raise ValueError(f"Linear layer '{data.get('id', 'unknown')}' must have a 'weightTying' field")
        
        layer = nn.Linear(
            in_features=data["inDim"],
            out_features=data["outDim"],
            bias=data.get("bias"),
            dtype=dtype,  # dtype 추가
        )
        
        # 🔑 이후 tying 단계에서 인식할 수 있도록 플래그와 메타정보 부착
        layer._weight_tying = bool(data.get("weightTying"))   # <-- 추가
        layer._declared_inDim = data["inDim"]                 # <-- 추가
        layer._declared_outDim = data["outDim"]               # <-- 추가
        return layer


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
            #    RotaryPositionalEmbedding은 제외 (RoPE는 어텐션 내부에서 처리됨)
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

    def forward_cached(self, x, caches=None, start_pos=0, use_cache=True):
        """캐시를 사용하여 레이어를 포워드 (Llama2 형식)"""
        if caches is None: caches = {}
        new_caches = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, (MultiHeadAttentionUsingSDP, GroupedQueryAttention)):
                out, new_cache = layer(x, start_pos=start_pos, kv_cache=caches.get(i), use_cache=use_cache, return_cache=True)
                x = out
                new_caches[i] = new_cache
            else:
                # 기존 토큰/상대/학습형 포지셔널 임베딩 합산 로직은 유지
                x = layer(x)
        return x, new_caches


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
    for node in layer_nodes:
        print(f"Creating layer: {node['data']['id']}")
        layer = LayerFactory.create_layer(node, dtype=torch_dtype)
        if "id" in node["data"]:
            layer.layer_id = node["data"]["id"]
            id_to_module[layer.layer_id] = layer
        layers.append(layer)

    # === Weight tying 단계 추가 ===
    # 1) 기준이 될 TokenEmbedding을 찾음 (가장 먼저/마지막으로 등장한 것을 선택)
    token_emb = None
    for m in layers:
        if isinstance(m, TokenEmbedding):
            token_emb = m   # 여러 개면 마지막 것을 사용

    if token_emb is not None:
        # TokenEmbedding 내부 weight 접근 (TokenEmbedding.weight 또는 TokenEmbedding.embedding.weight 케이스 모두 처리)
        emb_w = getattr(token_emb, "weight", None)
        if emb_w is None and hasattr(token_emb, "embedding"):
            emb_w = getattr(token_emb.embedding, "weight", None)

        if emb_w is None:
            raise RuntimeError("TokenEmbedding weight not found for tying.")

        vocab_size, emb_dim = emb_w.shape

        for m in layers:
            if isinstance(m, torch.nn.Linear) and getattr(m, "_weight_tying", False):
                # 크기 검증
                if m._declared_inDim != emb_dim or m._declared_outDim != vocab_size:
                    raise ValueError(
                        f"Weight tying size mismatch: Linear(in={m._declared_inDim}, out={m._declared_outDim}) "
                        f"vs Embedding(vocab={vocab_size}, emb={emb_dim})"
                    )
                # 실제 tying: 같은 Parameter를 참조하게 함
                m.weight = emb_w
                print(f"--- Weight tying: {m.layer_id} ---")

                # (선택) lm_head bias를 쓰지 않도록 권장
                # if m.bias is not None:
                #     with torch.no_grad():
                #         m.bias.zero_()
    else:
        # 모델 안에 TokenEmbedding이 없는데 weightTying True인 Linear가 있으면 에러로 안내
        for m in layers:
            if isinstance(m, torch.nn.Linear) and getattr(m, "_weight_tying", False):
                raise ValueError("Linear(weightTying=True) found but no TokenEmbedding layer exists.")

    return CustomSequential(layers, id_to_module)

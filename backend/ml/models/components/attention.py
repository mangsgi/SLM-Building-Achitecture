import math
import torch
import torch.nn as nn


# ------------------------------
# Shared buffers (mask, RoPE cos/sin) with stable cache keys
# ------------------------------
class SharedBuffers:
    _buffers = {}

    @staticmethod
    def get_buffers(context_length, head_dim, rope_base, freq_config, dtype=torch.float32):
        # dict 순서에 의존하지 않도록 key를 정규화
        freq_key = tuple(sorted(freq_config.items())) if freq_config else None
        key = (context_length, head_dim, rope_base, freq_key, dtype)

        if key not in SharedBuffers._buffers:
            # 마스크는 bool로 만들어 메모리 절약
            mask = torch.triu(
                torch.ones(context_length, context_length, dtype=torch.bool),
                diagonal=1
            )
            cos, sin = precompute_rope_params(head_dim, rope_base, context_length, freq_config)
            if dtype is not None:
                cos = cos.to(dtype)
                sin = sin.to(dtype)
            SharedBuffers._buffers[key] = (mask, cos, sin)

        return SharedBuffers._buffers[key]


# ------------------------------
# RoPE utilities
# ------------------------------
def precompute_rope_params(head_dim, rope_base, context_length, freq_config=None):
    """
    cos/sin은 (seq_len, head_dim/2) 형태로 반환한다.
    """
    if freq_config is None:
        freq_config = {"type": "default"}

    if freq_config["type"] == "default":
        # inv_freq: (head_dim/2,)
        inv_freq = 1.0 / (rope_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        # t: (seq_len,)
        t = torch.arange(context_length, dtype=inv_freq.dtype, device=inv_freq.device)
        # freqs: (seq_len, head_dim/2)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos()  # (seq_len, head_dim/2)
        sin = freqs.sin()  # (seq_len, head_dim/2)
    else:
        raise ValueError(f"Unknown frequency config type: {freq_config['type']}")

    return cos, sin


def compute_rope(x, cos, sin):
    """
    RoPE(Rotary Position Embedding) 적용.
    x:   (batch, num_heads, seq_len, head_dim), head_dim은 짝수
    cos: (seq_len, head_dim/2)
    sin: (seq_len, head_dim/2)
    """
    head_dim = x.size(-1)
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")

    # 실수부/허수부 분할
    x1, x2 = x.chunk(2, dim=-1)  # (..., head_dim/2) each

    # 브로드캐스팅을 위한 차원 확장: (1, 1, seq_len, head_dim/2)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # 회전 적용
    x_rope = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rope


# ------------------------------
# 1) 전형적 MHA (QKV 통합 프로젝션, 캐주얼 마스크)
# ------------------------------
class MultiHeadAttentionCombinedQKV(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout=0.0, qkv_bias=False, dtype=torch.float32):
        super().__init__()
        assert d_out % num_heads == 0, "embed_dim is indivisible by num_heads"

        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias, dtype=dtype)
        self.proj = nn.Linear(d_out, d_out, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

        # 캐주얼 마스크 (bool)
        mask = torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        # (b, T, E) -> (b, T, 3E)
        qkv = self.qkv(x)

        # (b, T, 3E) -> (b, T, 3, H, D)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, T, 3, H, D) -> (3, b, H, T, D)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # 3개 텐서로 분리
        queries, keys, values = qkv.unbind(0)  # each: (b, H, T, D)

        # scaled dot product
        attn_scores = queries @ keys.transpose(-2, -1)  # (b, H, T, T)

        # 안전한 마스킹 값 (dtype별 최소값)
        fill_value = torch.finfo(attn_scores.dtype).min
        causal = self.mask[:num_tokens, :num_tokens]  # (T, T)
        attn_scores = attn_scores.masked_fill(causal, fill_value)

        # 정석 스케일링: / sqrt(D)
        attn_scores = attn_scores / math.sqrt(self.head_dim)

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # (b, H, T, T) @ (b, H, T, D) -> (b, H, T, D)
        context_vec = attn_weights @ values

        # (b, H, T, D) -> (b, T, H, D) -> (b, T, E)
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.num_heads * self.head_dim)
        context_vec = self.proj(context_vec)
        return context_vec


# ------------------------------
# 2) Grouped-Query Attention (GQA) + RoPE
# ------------------------------
class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        context_length,
        num_heads,
        num_kv_groups,      # GQA 그룹 수
        rope_base=10_000,   # RoPE 기본값
        rope_config=None,   # RoPE 추가 설정 (dict)
        dropout=0.0,
        dtype=torch.float32,
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        # GQA: Q는 H개, K/V는 그룹당 1개
        self.W_query = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.W_key   = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)

        self.attn_dropout = nn.Dropout(dropout)

        # SharedBuffers에서 받아와 버퍼로 등록(모듈과 함께 .to(device) 이동)
        mask, cos, sin = SharedBuffers.get_buffers(context_length, self.head_dim, rope_base, rope_config, dtype)
        self.register_buffer("mask", mask)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x):
        batch_size, num_tokens, _ = x.shape

        # Q, K, V 계산
        queries = self.W_query(x)  # (b, T, d_out)
        keys    = self.W_key(x)    # (b, T, G * D)
        values  = self.W_value(x)  # (b, T, G * D)

        # reshape
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        keys    = keys.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim)
        values  = values.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim)

        # (b, T, H, D) -> (b, H, T, D), (b, T, G, D) -> (b, G, T, D)
        queries = queries.transpose(1, 2)
        keys    = keys.transpose(1, 2)
        values  = values.transpose(1, 2)

        # RoPE (Q, K에만)
        keys    = compute_rope(keys, self.cos, self.sin)
        queries = compute_rope(queries, self.cos, self.sin)

        # GQA: K/V를 group_size만큼 복제하여 H개에 매핑
        keys   = keys.repeat_interleave(self.group_size, dim=1)   # (b, H, T, D)
        values = values.repeat_interleave(self.group_size, dim=1) # (b, H, T, D)

        # 점수/마스킹/스케일링
        attn_scores = queries @ keys.transpose(-2, -1)  # (b, H, T, T)
        fill_value = torch.finfo(attn_scores.dtype).min
        causal = self.mask[:num_tokens, :num_tokens]
        attn_scores = attn_scores.masked_fill(causal, fill_value)
        attn_scores = attn_scores / math.sqrt(self.head_dim)

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 컨텍스트
        context_vec = attn_weights @ values  # (b, H, T, D)
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec


# ------------------------------
# 3) PyTorch Scaled-Dot-Product Attention (SDPA) 백엔드
# ------------------------------
class MHAPyTorchScaledDotProduct(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout=0.0, qkv_bias=False, is_rope=False, theta=10000.0, dtype=torch.float32):
        super().__init__()
        assert d_out % num_heads == 0, "embed_dim is indivisible by num_heads"

        self.is_rope = is_rope
        self.theta = float(theta)
        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias, dtype=dtype)
        self.proj = nn.Linear(d_out, d_out, dtype=dtype)
        self.dropout = dropout  # float. SDPA에 전달
        
        # RoPE 준비(필요할 때만)
        if self.is_rope:
            if self.head_dim % 2 != 0:
                raise ValueError(f"RoPE requires even head_dim, got {self.head_dim}.")
            # mask는 SDPA에서 is_causal=True로 대체되므로 cos/sin만 사용
            _mask, cos, sin = SharedBuffers.get_buffers(
                self.context_length, self.head_dim, self.theta,
                freq_config={"type": "default"}, dtype=dtype
            )
            self.register_buffer("cos", cos)  # (ctx_len, D/2)
            self.register_buffer("sin", sin)  # (ctx_len, D/2)

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape
        if self.is_rope and num_tokens > self.context_length:
            # 버퍼 범위를 넘으면 명확히 실패(원하면 여기서 버퍼 재계산 로직 추가 가능)
            raise ValueError(
                f"Sequence length {num_tokens} exceeds RoPE buffer (context_length={self.context_length})."
            )

        # (b, T, E) -> (b, T, 3E)
        qkv = self.qkv(x)

        # (b, T, 3E) -> (b, T, 3, H, D)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, T, 3, H, D) -> (3, b, H, T, D)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # 3개 텐서로 분리
        queries, keys, values = qkv.unbind(0)  # each: (b, H, T, D)

        # ★ RoPE: Q, K에만 적용
        if self.is_rope:
            cos = self.cos[:num_tokens]  # (T, D/2)
            sin = self.sin[:num_tokens]  # (T, D/2)
            queries = compute_rope(queries, cos, sin)
            keys    = compute_rope(keys,    cos, sin)

        # 학습 중일 때만 드롭아웃 적용
        use_dropout = 0.0 if not self.training else self.dropout

        # PyTorch 2.x SDPA: 내부에서 스케일/마스킹/최적화(플래시/트라이턴)까지 처리
        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values,
            attn_mask=None,
            dropout_p=use_dropout,
            is_causal=True
        )  # (b, H, T, D)

        # (b, H, T, D) -> (b, T, H, D) -> (b, T, E)
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)
        context_vec = self.proj(context_vec)
        return context_vec

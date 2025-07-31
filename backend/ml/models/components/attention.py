import torch
import torch.nn as nn
import math

class SharedBuffers:
    _buffers = {}

    @staticmethod
    def get_buffers(context_length, head_dim, rope_base, freq_config, dtype=torch.float32):
        key = (context_length, head_dim, rope_base, tuple(freq_config.values()) if freq_config else freq_config, dtype)

        if key not in SharedBuffers._buffers:
            # Create or fetch the buffers
            mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
            cos, sin = precompute_rope_params(head_dim, rope_base, context_length, freq_config)
            if dtype is not None:
                cos = cos.to(dtype)
                sin = sin.to(dtype)
            SharedBuffers._buffers[key] = (mask, cos, sin)

        return SharedBuffers._buffers[key]

def precompute_rope_params(head_dim, rope_base, context_length, freq_config=None):
    if freq_config is None:
        freq_config = {"type": "default"}
    
    if freq_config["type"] == "default":
        # Default RoPE implementation
        inv_freq = 1. / (rope_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(context_length, device=inv_freq.device).type_as(inv_freq)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
    else:
        raise ValueError(f"Unknown frequency config type: {freq_config['type']}")
    
    return cos, sin

def compute_rope(x, cos, sin):
    """
    RoPE(Rotary Position Embedding)를 적용합니다.
    x: (batch_size, num_heads, seq_len, head_dim)
    cos, sin: (seq_len, head_dim)
    """
    # head_dim이 2의 배수인지 확인
    head_dim = x.size(-1)
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    
    # x를 실수부와 허수부로 분할
    x1, x2 = x.chunk(2, dim=-1)
    
    # cos, sin을 적절한 shape로 변환
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim/2)
    sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim/2)
    
    # RoPE 적용
    x_rope = torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1)
    
    return x_rope

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

        # Create causal mask
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_head, num_tokens, head_dim)
        queries, keys, values = qkv.unbind(0)

        # (b, num_heads, num_tokens, head_dim) --> (b, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(-2, -1)
        attn_scores = attn_scores.masked_fill(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**-0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # (b, num_heads, num_tokens, num_tokens) --> (b, num_heads, num_tokens, head_dim)
        context_vec = attn_weights @ values

        # (b, num_heads, num_tokens, head_dim) --> (b, num_tokens, num_heads, head_dim)
        context_vec = context_vec.transpose(1, 2)

        # (b, num_tokens, num_heads, head_dim) --> (b, num_tokens, embed_dim)
        context_vec = context_vec.contiguous().view(batch_size, num_tokens, embed_dim)

        context_vec = self.proj(context_vec)

        return context_vec


class GroupedQueryAttention(nn.Module):
    def __init__(
            self, d_in, d_out, context_length, num_heads,
            num_kv_groups,       # GQA를 위한 파라미터
            rope_base=10_000,    # RoPE 기본값
            rope_config=None,    # RoPE 추가 설정
            dtype=torch.float32
        ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        # GQA를 위한 분리된 선형 변환
        self.W_query = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)

        # Fetch buffers using SharedBuffers
        mask, cos, sin = SharedBuffers.get_buffers(context_length, self.head_dim, rope_base, rope_config, dtype)
        self.register_buffer("mask", mask)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x):
        batch_size, num_tokens, d_in = x.shape

        # GQA 방식으로 QKV 계산
        queries = self.W_query(x)  # (b, num_tokens, d_out)
        keys = self.W_key(x)       # (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)   # (b, num_tokens, num_kv_groups * head_dim)

        # Reshape
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim)

        # Transpose
        queries = queries.transpose(1, 2)  # (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)        # (b, num_kv_groups, num_tokens, head_dim)
        values = values.transpose(1, 2)    # (b, num_kv_groups, num_tokens, head_dim)

        # Apply RoPE
        keys = compute_rope(keys, self.cos, self.sin)
        queries = compute_rope(queries, self.cos, self.sin)

        # Expand keys and values for GQA
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        # Attention 계산
        attn_scores = queries @ keys.transpose(-2, -1)
        attn_scores = attn_scores.masked_fill(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**-0.5, dim=-1)

        # Context vector 계산
        context_vec = attn_weights @ values
        context_vec = context_vec.transpose(1, 2).contiguous()
        context_vec = context_vec.view(batch_size, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec


class MHAPyTorchScaledDotProduct(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout=0.0, qkv_bias=False, dtype=torch.float32):
        super().__init__()
        assert d_out % num_heads == 0, "embed_dim is indivisible by num_heads"

        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias, dtype=dtype)
        self.proj = nn.Linear(d_out, d_out, dtype=dtype)
        self.dropout = dropout

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv

        use_dropout = 0.0 if not self.training else self.dropout

        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, 
            attn_mask=None, 
            dropout_p=use_dropout, 
            is_causal=True
        )

        # Combine heads
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)
        context_vec = self.proj(context_vec)

        return context_vec
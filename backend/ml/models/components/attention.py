import math
import torch
import torch.nn as nn
from typing import Optional

from .normalization import RMSNorm



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
    - default: 전통적인 RoPE inv_freq
    - llama3_scaled: rope_freq(=freq_config)로 주파수를 구간별/스무딩 스케일
    """
    assert head_dim % 2 == 0, "head_dim must be even"
    device = torch.device("cpu")
    dtype = torch.float32

    # base inv_freq (D/2,)
    inv_freq = 1.0 / (rope_base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))

    # detect llama3-style config
    use_llama3 = False
    if isinstance(freq_config, dict):
        if freq_config.get("type") == "llama3_scaled":
            use_llama3 = True
        else:
            # 키가 그대로 넘어오는 경우(factor/low_freq_factor/high_freq_factor/original_context_length)
            needed = {"factor", "low_freq_factor", "high_freq_factor", "original_context_length"}
            if needed.issubset(set(freq_config.keys())):
                use_llama3 = True

    if use_llama3:
        # 파라미터 읽기
        factor = float(freq_config["factor"])
        low_f  = float(freq_config["low_freq_factor"])
        high_f = float(freq_config["high_freq_factor"])
        L0     = float(freq_config["original_context_length"])

        # 파장(wavelength) 기반 스무딩 스케일
        # wavelen = 2*pi / inv_freq
        wavelen = 2 * torch.pi / inv_freq  # (D/2,)

        low_wavelen  = L0 / low_f
        high_wavelen = L0 / high_f

        # 세 영역: 저주파(> low), 중간 빈도([high, low]), 고주파(< high)
        inv_freq_scaled = inv_freq.clone()

        # 저주파: 전부 factor로 나눔
        low_mask = wavelen > low_wavelen
        inv_freq_scaled = torch.where(low_mask, inv_freq / factor, inv_freq_scaled)

        # 중간: 선형 스무딩
        mid_mask = (wavelen <= low_wavelen) & (wavelen >= high_wavelen)
        # smooth_factor \in [0,1]로 보정
        smooth = (L0 / wavelen - low_f) / (high_f - low_f)
        smooth = torch.clamp(smooth, 0.0, 1.0)
        smoothed = (1.0 - smooth) * (inv_freq / factor) + smooth * inv_freq
        inv_freq_scaled = torch.where(mid_mask, smoothed, inv_freq_scaled)

        # 고주파 구간은 원래 inv_freq 유지
        inv_freq = inv_freq_scaled

    # positions (T,)
    t = torch.arange(context_length, device=inv_freq.device, dtype=inv_freq.dtype)

    # (T, D/2)
    freqs = torch.einsum("i,j->ij", t, inv_freq)

    # cos/sin (T, D/2)
    cos = freqs.cos()
    sin = freqs.sin()
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
# 1) Multi-Head Attention Using Scaled-Dot-Product Attention (SDPA) 백엔드
# ------------------------------
class MultiHeadAttentionUsingSDP(nn.Module):
    def __init__(self, 
                 d_in, d_out, num_heads, context_length, 
                 dropout=0.1, qkv_bias=True, proj_bias=None, is_rope=False, rope_base=10000.0, dtype=torch.float32):
        super().__init__()
        assert d_out % num_heads == 0, "embed_dim is indivisible by num_heads"

        self.is_rope = is_rope
        self.rope_base = float(rope_base)
        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias, dtype=dtype)
        
        if proj_bias is None:
            proj_bias = qkv_bias
        self.proj = nn.Linear(d_out, d_out, bias=proj_bias, dtype=dtype)
        self.dropout = dropout  # float. SDPA에 전달
        
        # RoPE 준비(필요할 때만)
        if self.is_rope:
            if self.head_dim % 2 != 0:  
                raise ValueError(f"RoPE requires even head_dim, got {self.head_dim}.")
            # mask는 SDPA에서 is_causal=True로 대체되므로 cos/sin만 사용
            _mask, cos, sin = SharedBuffers.get_buffers(
                self.context_length, self.head_dim, self.rope_base,
                freq_config={"type": "default"}, dtype=dtype
            )
            self.register_buffer("cos", cos)  # (ctx_len, D/2)
            self.register_buffer("sin", sin)  # (ctx_len, D/2)

    def forward(self, x, start_pos: int = 0, kv_cache=None, use_cache: bool = False, return_cache: bool = False):
        batch_size, num_tokens, embed_dim = x.shape
        if self.is_rope and (start_pos + num_tokens) > self.context_length:            
            # 버퍼 범위를 넘으면 명확히 실패(원하면 여기서 버퍼 재계산 로직 추가 가능)
            raise ValueError(
                f"Sequence length {start_pos + num_tokens} exceeds RoPE buffer (context_length={self.context_length})."
            )

        qkv = self.qkv(x) # (b, T, E) -> (b, T, 3E)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim) # (b, T, 3E) -> (b, T, 3, H, D)
        qkv = qkv.permute(2, 0, 3, 1, 4) # (b, T, 3, H, D) -> (3, b, H, T, D)

        # 3개 텐서로 분리
        queries, keys, values = qkv.unbind(0)  # each: (b, H, T, D)

        # ★ RoPE: Q, K에만 적용
        if self.is_rope:
            cos = self.cos[start_pos : start_pos + num_tokens] # (T, D/2)
            sin = self.sin[start_pos : start_pos + num_tokens] # (T, D/2)
            queries = compute_rope(queries, cos, sin)
            keys    = compute_rope(keys,    cos, sin)

        # 캐시 결합
        if use_cache and kv_cache is not None:
            keys = torch.cat([kv_cache[0], keys], dim=2)
            values = torch.cat([kv_cache[1], values], dim=2)
        else:
            pass
        
        # context_length 초과 시 꼬리만 유지
        if keys.size(2) > self.context_length:
            keys   = keys[:, :, -self.context_length:, :]
            values = values[:, :, -self.context_length:, :]

        # 학습 중일 때만 드롭아웃 적용
        use_dropout = self.dropout if (self.training and self.dropout > 0.0) else 0.0

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
        
        if return_cache:
            return context_vec, (keys, values)
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
        num_kv_groups,                  # GQA 그룹 수
        head_dim: Optional[int] = None,
        rope_base=500_000.0,            # RoPE 기본값
        rope_config=None,               # RoPE 추가 설정 (dict)
        dropout=0.0,                    # attention dropout 확률로 사용
        dtype=torch.float32,
        is_rope: bool = True,           # RoPE 사용 여부(범용성을 위해 옵션화)
        qkv_bias: bool = False,         # Q/K/V에 bias 사용할지
        out_proj_bias: bool = False,    # 출력 프로젝션 bias
        qk_norm: bool = False,          # "rms" | "layer" | None
        qk_norm_eps: float = 1e-6,
    ):
        super().__init__()
        # assert d_out % num_heads == 0, "d_out must be divisible by num_heads" # Qwen3처럼 num_heads * head_dim != d_out인 모델도 고려
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"
        
        self.d_out = d_out
        self.head_dim = head_dim if head_dim is not None else (d_out // num_heads)
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups
        self.context_length = context_length

        # RoPE는 선택 사항. 사용할 때만 짝수 차원 요구.
        self.use_rope = bool(is_rope)
        if self.use_rope and (self.head_dim % 2 != 0):
            raise ValueError("RoPE requires even head_dim.")

        # Q: (E -> H*D), K/V: (E -> G*D), out: (H*D -> E) / Q는 H개, K/V는 그룹당 1개
        self.W_query = nn.Linear(d_in, self.num_heads * self.head_dim, bias=qkv_bias, dtype=dtype)
        self.W_key   = nn.Linear(d_in, self.num_kv_groups * self.head_dim, bias=qkv_bias, dtype=dtype)
        self.W_value = nn.Linear(d_in, self.num_kv_groups * self.head_dim, bias=qkv_bias, dtype=dtype)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, d_out, bias=out_proj_bias, dtype=dtype)

        # attention dropout (훈련 시에만 작동; residual dropout은 블록 바깥에서 처리)
        self.attn_dropout = nn.Dropout(dropout)

        # 마스크는 직접 만든다 (항상 필요)
        mask = torch.triu(torch.ones(self.context_length, self.context_length, dtype=torch.bool), diagonal=1)
        self.register_buffer("mask", mask)

        # RoPE가 켜진 경우에만 cos/sin 준비
        if self.use_rope:
            rope_cfg = {"type": "llama3_scaled", **(rope_config or {})} if rope_config else {"type": "default"}
            _mask, cos, sin = SharedBuffers.get_buffers(self.context_length, self.head_dim, rope_base, rope_cfg, dtype)
            self.register_buffer("cos", cos)
            self.register_buffer("sin", sin)
 
        # QK 정규화
        self.qk_norm = qk_norm
        self.qk_norm_eps = qk_norm_eps
        if self.qk_norm:
            # 마지막 차원(head_dim) 기준 RMSNorm
            self.q_norm = RMSNorm(self.head_dim, eps=qk_norm_eps, dtype=dtype)
            self.k_norm = RMSNorm(self.head_dim, eps=qk_norm_eps, dtype=dtype)
        else:
            self.q_norm = self.k_norm = None    
        

    @torch.no_grad()
    def _build_causal_mask(self, T_q: int, T_k: int, device: torch.device):
        """
        T_q x T_k 마스크를 생성.
        - 캐시가 없으면 (T_q == T_k) 이고, self.mask[:T_q, :T_q] 사용.
        - 캐시가 있으면 마지막 T_q x T_q 블록에만 상삼각 마스크를 적용.
        """
        if T_k == T_q:
            return self.mask[:T_q, :T_q]  # (T, T) bool

        # 일반화된 마스크: [zeros(T, T_k - T_q) | upper-tri(T, T_q)]
        m = torch.zeros(T_q, T_k, dtype=torch.bool, device=device)
        m[:, -T_q:] = self.mask[:T_q, :T_q]
        return m  # (T_q, T_k) bool

    def forward(self, x, start_pos: int = 0, kv_cache=None, use_cache: bool = False, return_cache: bool = False):
        """
        x: (B, T, E)
        kv_cache: (k_cache_g, v_cache_g) where k_cache_g/v_cache_g are (B, G, T_cache, D)
        """
        batch_size, num_tokens, _ = x.shape
        if self.use_rope and (start_pos + num_tokens) > self.context_length:
            raise ValueError(
                f"Sequence end {start_pos + num_tokens} exceeds RoPE buffer (context_length={self.context_length})."
            )

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

        # RoPE (Q, K에만) — 선택 적용
        if self.use_rope:
            cos = self.cos[start_pos : start_pos + num_tokens]  # (T, D/2)
            sin = self.sin[start_pos : start_pos + num_tokens]  # (T, D/2)
            keys    = compute_rope(keys, cos, sin)
            queries = compute_rope(queries, cos, sin)

        # QK-Norm (per-head, pre-dot-product)
        if self.q_norm is not None:
            # (B, H(or G), T, D) → head_dim 기준 정규화
            queries = self.q_norm(queries)
        if self.k_norm is not None:
            keys    = self.k_norm(keys)

        # (변수 이름을 분리해 두는 게 안전)
        k_g = keys
        v_g = values

        # 캐시 결합 (그룹 기준)
        if use_cache and kv_cache is not None:
            k_cache_g, v_cache_g = kv_cache              # (B, G, T_cache, D)
            k_g = torch.cat([k_cache_g, k_g], dim=2)     # (B, G, T_k, D)
            v_g = torch.cat([v_cache_g, v_g], dim=2)

        # 컨텍스트 길이 초과 시 꼬리만 유지 (그룹 기준에서)
        if k_g.size(2) > self.context_length:
            k_g = k_g[:, :, -self.context_length:, :]
            v_g = v_g[:, :, -self.context_length:, :]

        # GQA: K/V를 group_size만큼 복제하여 H개에 매핑
        keys_h   = k_g.repeat_interleave(self.group_size, dim=1)   # (b, H, T, D)
        values_h = v_g.repeat_interleave(self.group_size, dim=1)   # (b, H, T, D)
        T_k = keys_h.size(2)

        # 점수/마스킹/스케일링
        attn_scores = torch.matmul(queries, keys_h.transpose(-2, -1))   # (b, H, T, T_k)
        attn_scores = attn_scores / math.sqrt(self.head_dim)

        # 일반화된 캐주얼 마스크 생성 및 적용
        causal = self._build_causal_mask(num_tokens, T_k, device=attn_scores.device)  # (T, T_k) bool
        attn_scores = attn_scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), torch.finfo(attn_scores.dtype).min)

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)  # eval()에서는 자동 no-op

        # 컨텍스트 & 출력
        context = torch.matmul(attn_weights, values_h)  # (B, H, T, D)
        context = context.transpose(1, 2).contiguous()  # (B, T, H, D)
        # Qwen3처럼 num_heads * head_dim != d_out인 모델도 고려
        context = context.view(batch_size, num_tokens, self.num_heads * self.head_dim) # (B, T, H * D)
        out = self.out_proj(context)

        if return_cache:
            # 그룹 기준 캐시를 반환해야 메모리 효율적
            return out, (k_g, v_g)
        return out





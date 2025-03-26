import torch
import torch.nn as nn

from config import GPT_CONFIG_124M as config

torch.manual_seed(123)

batch_size = config["batch_size"]
emb_dim = config["emb_dim"]
context_length = config["context_length"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embeddings = torch.randn((batch_size, context_length, emb_dim), device=device)



class MultiHeadAttentionCombinedQKV(nn.Module): # 일반적인 multi head attention + QKV를 하나의 선형층으로 계산 -> 더 효율적
    def __init__(self, d_in, d_out, num_heads, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()

        assert d_out % num_heads == 0, "embed_dim is indivisible by num_heads"

        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

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

def mha_combined():
    mha_combined_qkv = MultiHeadAttentionCombinedQKV(
        d_in=emb_dim,
        d_out=emb_dim,
        context_length=context_length,
        dropout=0.0,
        num_heads=12,
        qkv_bias=False
    ).to(device)

    out = mha_combined_qkv(embeddings)
    print(out.shape)
    
    mha_combined_total_params = sum(p.numel() for p in mha_combined_qkv.parameters())
    print(f"MHA_combined: {mha_combined_total_params:,}")


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


# class GroupedQueryAttention(nn.Module):
#     def __init__(
#             self, d_in, d_out, context_length, num_heads,
#             num_kv_groups,       # NEW
#             rope_base=10_000,    # NEW
#             rope_config=None,    # NEW
#             dtype=None
#         ):
#         super().__init__()
#         assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
#         assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"  # NEW

#         self.d_out = d_out
#         self.num_heads = num_heads
#         self.head_dim = d_out // num_heads

#         ############################# NEW  #############################
#         # self.W_key = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
#         # self.W_value = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
#         self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
#         self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
#         self.num_kv_groups = num_kv_groups
#         self.group_size = num_heads // num_kv_groups
#         ################################################################

#         self.W_query = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
#         self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)

#         ############################# NEW  #############################
#         # Fetch buffers using SharedBuffers
#         mask, cos, sin = SharedBuffers.get_buffers(context_length, self.head_dim, rope_base, rope_config, dtype)
#         ############################# NEW  #############################
        
#         self.register_buffer("mask", mask)
#         self.register_buffer("cos", cos)
#         self.register_buffer("sin", sin)

#     def forward(self, x):
#         b, num_tokens, d_in = x.shape

#         queries = self.W_query(x)  # Shape: (b, num_tokens, d_out)
#         keys = self.W_key(x)  # Shape: (b, num_tokens, num_kv_groups * head_dim)
#         values = self.W_value(x)  # Shape: (b, num_tokens, num_kv_groups * head_dim)

#         # Reshape queries, keys, and values
#         queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

#         ##################### NEW  #####################
#         # keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
#         # values = values.view(b, num_tokens, self.num_heads, self.head_dim)
#         keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim)
#         values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim)
#         ################################################

#         # Transpose keys, values, and queries
#         keys = keys.transpose(1, 2)  # Shape: (b, num_heads, num_tokens, head_dim)
#         values = values.transpose(1, 2)  # Shape: (b, num_heads, num_tokens, head_dim)
#         queries = queries.transpose(1, 2)  # Shape: (b, num_query_groups, num_tokens, head_dim)

#         # Apply RoPE
#         keys = compute_rope(keys, self.cos, self.sin)
#         queries = compute_rope(queries, self.cos, self.sin)

#         ##################### NEW  #####################
#         # Expand keys and values to match the number of heads
#         # Shape: (b, num_heads, num_tokens, head_dim)

#         keys = keys.repeat_interleave(self.group_size, dim=1)  # Shape: (b, num_heads, num_tokens, head_dim)
#         values = values.repeat_interleave(self.group_size, dim=1)  # Shape: (b, num_heads, num_tokens, head_dim)
#         # For example, before repeat_interleave along dim=1 (query groups):
#         #   [K1, K2]
#         # After repeat_interleave (each query group is repeated group_size times):
#         #   [K1, K1, K2, K2]
#         # If we used regular repeat instead of repeat_interleave, we'd get:
#         #   [K1, K2, K1, K2]
#         ################################################

#         # Compute scaled dot-product attention (aka self-attention) with a causal mask
#         # Shape: (b, num_heads, num_tokens, num_tokens)
#         attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

#         # Original mask truncated to the number of tokens and converted to boolean
#         mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

#         # Use the mask to fill attention scores
#         attn_scores.masked_fill_(mask_bool, -torch.inf)

#         attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
#         assert keys.shape[-1] == self.head_dim

#         # Shape: (b, num_tokens, num_heads, head_dim)
#         context_vec = (attn_weights @ values).transpose(1, 2)

#         # Combine heads, where self.d_out = self.num_heads * self.head_dim
#         context_vec = context_vec.reshape(b, num_tokens, self.d_out)
#         context_vec = self.out_proj(context_vec)  # optional projection

#         return context_vec

# def grouped_query():
#     gqa = GroupedQueryAttention(
#         d_in=emb_dim,
#         d_out=emb_dim,
#         context_length=context_length,
#         num_heads=32,
#         num_kv_groups=8,
#         rope_base=llama_3_theta_base
#     ).to(device)

#     out = gqa(embeddings)
#     print(out.shape)
    # gqa_total_params = sum(p.numel() for p in gqa.parameters())
    # print(f"GQA: {gqa_total_params:,}")




class MHAPyTorchScaledDotProduct(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()

        assert d_out % num_heads == 0, "embed_dim is indivisible by num_heads"

        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
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

        use_dropout = 0. if not self.training else self.dropout

        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=use_dropout, is_causal=True)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)

        context_vec = self.proj(context_vec)

        return context_vec

def mha_flash():
    mha_pytorch_scaled = MHAPyTorchScaledDotProduct(
        d_in=emb_dim,
        d_out=emb_dim,
        context_length=context_length,
        dropout=0.0,
        num_heads=12,
        qkv_bias=False
    ).to(device)

    out = mha_pytorch_scaled(embeddings)
    print(out.shape)
    mha_pytorch_scaled_total_params = sum(p.numel() for p in mha_pytorch_scaled.parameters())
    print(f"Flash: {mha_pytorch_scaled_total_params:,}")


if __name__ == "__main__":
    mha_combined()
    # grouped_query()
    mha_flash()




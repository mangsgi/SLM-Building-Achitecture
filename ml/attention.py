import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model은 num_heads로 나누어 떨어져야 함!"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads  # 각 헤드의 차원
        self.scale = self.head_dim ** 0.5  # 정규화

        # Q, K, V를 위한 선형 변환
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        # Q, K, V 추출 후 (batch, seq_len, num_heads, head_dim) 형태로 변환
        qkv = self.qkv_proj(x).reshape(batch_size, seq_len, self.num_heads, 3 * self.head_dim)
        Q, K, V = torch.chunk(qkv, 3, dim=-1)  # (batch, seq_len, num_heads, head_dim)

        # 어텐션 점수 계산
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, num_heads, seq_len, seq_len)
        attn_weights = torch.softmax(scores, dim=-1)

        # 가중치를 적용한 V 계산
        output = torch.matmul(attn_weights, V)  # (batch, num_heads, seq_len, head_dim)
        output = output.reshape(batch_size, seq_len, d_model)  # 다시 합치기

        return self.output_proj(output)

# 테스트
x = torch.rand(2, 10, 512)  # (batch, seq_len, d_model)
mhsa = MultiHeadSelfAttention(d_model=512, num_heads=8)
output = mhsa(x)
print(output.shape)  # torch.Size([2, 10, 512])

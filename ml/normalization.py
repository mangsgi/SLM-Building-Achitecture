import torch
import torch.nn as nn

from config import GPT_CONFIG_124M as config


emb_dim = config["emb_dim"]
x = torch.rand(2, 10, 768)  # (batch, seq_len, d_model)


# 1. 모든 피처에 대해 평균과 분산 계산하여 정규화
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))  # 학습 가능한 스케일 파라미터
        self.shift = nn.Parameter(torch.zeros(d_model))  # 학습 가능한 이동 파라미터
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  # 마지막 차원 기준 평균
        var = x.var(dim=-1, unbiased=False, keepdim=True)  # 분산 계산
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

# 예제
def layer_ex():
    layer_norm = LayerNorm(emb_dim)
    output = layer_norm(x)
    print(output.shape)  # torch.Size([2, 10, 512])


# 2. 평균을 빼지 않고 분산만 정규화
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))  # 학습 가능한 스케일 파라미터
        self.eps = eps

    def forward(self, x):
        norm = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)  # Root Mean Square 계산
        return self.scale * (x / norm)

# 예제
def rms_ex():
    rms_norm = RMSNorm(emb_dim)
    output = rms_norm(x)
    print(output.shape)  # torch.Size([2, 10, 512])


if __name__ == "__main__":
    layer_ex()
    rms_ex()

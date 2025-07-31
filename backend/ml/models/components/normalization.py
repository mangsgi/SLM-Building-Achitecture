import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6, dtype=torch.float32):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model, dtype=dtype))
        self.shift = nn.Parameter(torch.zeros(d_model, dtype=dtype))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6, dtype=torch.float32):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model, dtype=dtype))
        self.eps = eps

    def forward(self, x):
        norm = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.scale * (x / norm)

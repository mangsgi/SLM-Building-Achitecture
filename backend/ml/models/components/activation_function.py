import torch
import torch.nn as nn
import torch.nn.functional as F

class ReLU(nn.Module):
    def forward(self, x):
        return F.relu(x)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class SiLU(nn.Module):  # Swish와 동일
    def forward(self, x):
        return F.silu(x)


class LeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return F.leaky_relu(x, negative_slope=self.negative_slope)


# 팩토리 맵 (새 인스턴스를 리턴)
activation_factory = {
    "relu":  lambda **kw: ReLU(),
    "gelu":  lambda **kw: GELU(),
    "silu":  lambda **kw: SiLU(),
    "swish": lambda **kw: SiLU(),  # 별칭
    "leaky": lambda **kw: LeakyReLU(kw.get("negative_slope", 0.01)),
}

# get_activation도 팩토리를 쓰도록 통일
def get_activation(name: str, **kwargs) -> nn.Module:
    key = name.lower()
    # 별칭 처리
    if key == "swish": key = "silu"
    if key not in activation_factory:
        raise ValueError(f"Unsupported activation function: {name}. "
                         f"Available: {', '.join(sorted(activation_factory))}")
    return activation_factory[key](**kwargs)

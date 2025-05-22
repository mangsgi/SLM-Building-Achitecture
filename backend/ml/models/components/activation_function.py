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


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return ReLU()
    elif name == "gelu":
        return GELU()
    elif name == "silu":
        return SiLU()
    elif name == "leaky":
        return LeakyReLU()
    else:
        raise ValueError(f"Unsupported activation function: {name}")


# activation_map은 선택적으로 사용
activation_map = {
    "relu": ReLU(),
    "gelu": GELU(),
    "silu": SiLU(),
    "leaky": LeakyReLU(),
}

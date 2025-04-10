import torch
import torch.nn as nn
import torch.nn.functional as F


class ReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.relu(x)


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
        """
        return F.gelu(x)


class SiLU(nn.Module):  # Swish와 동일
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        x * torch.sigmoid(x)
        """
        return F.silu(x)


class LeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return F.leaky_relu(x, negative_slope=self.negative_slope)
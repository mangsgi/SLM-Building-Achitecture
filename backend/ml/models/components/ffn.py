import torch
import torch.nn as nn

from .activation_function import activation_map, ReLU


class CustomFFN(nn.Module):
    def __init__(self, emb_dim, dff_ratio=4.0, activation="relu", gated=False, dtype=torch.float32):
        super().__init__()
        d_ff = int(emb_dim * dff_ratio)
        self.activation = activation_map.get(activation.lower(), ReLU())

        if gated:
            self.fc1 = nn.Linear(emb_dim, d_ff, dtype=dtype, bias=False)
            self.fc2 = nn.Linear(emb_dim, d_ff, dtype=dtype, bias=False)
            self.fc3 = nn.Linear(d_ff, emb_dim, dtype=dtype, bias=False)
            self.forward = self._gated_forward
        else:
            self.layers = nn.Sequential(
                nn.Linear(emb_dim, d_ff, dtype=dtype),
                self.activation,
                nn.Linear(d_ff, emb_dim, dtype=dtype),
            )
            self.forward = self._sequential_forward

    def _gated_forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = self.activation(x_fc1) * x_fc2
        return self.fc3(x)

    def _sequential_forward(self, x):
        return self.layers(x)


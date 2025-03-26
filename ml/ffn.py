import torch
import torch.nn as nn

from activation_function import *
from config import GPT_CONFIG_124M as config

torch.manual_seed(123)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
class CustomFFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_ff = int(cfg["emb_dim"] * cfg["dff_ratio"])

        self.activation = activation_map.get(cfg["activation"].lower(), ReLU())

        if cfg["gated"]:
            self.fc1 = nn.Linear(cfg["emb_dim"], d_ff, dtype=cfg["dtype"], bias=False)
            self.fc2 = nn.Linear(cfg["emb_dim"], d_ff, dtype=cfg["dtype"], bias=False)
            self.fc3 = nn.Linear(d_ff, cfg["emb_dim"], dtype=cfg["dtype"], bias=False)
            self.forward = self._gated_forward
        else:
            self.layers = nn.Sequential(
                nn.Linear(cfg["emb_dim"], d_ff),
                self.activation,
                nn.Linear(d_ff, cfg["emb_dim"]),
            )
            self.forward = self._sequential_forward

    def _gated_forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = self.activation(x_fc1) * x_fc2
        return self.fc3(x)

    def _sequential_forward(self, x):
        return self.layers(x)


def ffn():
    cfg = config
    ffn = CustomFFN(cfg)

    x = torch.randn(cfg["batch_size"], cfg["context_length"], cfg["emb_dim"], dtype=cfg["dtype"])

    out = ffn(x)

    print("입력 shape:", x.shape)   # torch.Size([2, 10, 512])
    print("출력 shape:", out.shape) # torch.Size([2, 10, 512])


if __name__ == "__main__":
    ffn()


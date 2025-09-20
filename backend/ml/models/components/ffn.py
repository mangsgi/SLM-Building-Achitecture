import torch
import torch.nn as nn

from .activation_function import get_activation


class CustomFFN(nn.Module):
    def __init__(self, emb_dim, hidden_dim=3072, activation="GELU", is_gated=False, dtype=torch.float32):
        super().__init__()
        act_key = activation.lower()

        # SWiGLU 문자열이 들어오면 자동 전환
        if act_key in {"swiglu", "swi-glu"}:
            is_gated = True #  SWiGLU는 무조건 Gated 방식
            act_key = "silu"  # SWiGLU의 게이트 비선형은 SiLU

        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.is_gated = is_gated
        self.act = get_activation(act_key)
        
        if is_gated:
            # W1, W3: emb_dim -> hidden_dim, W2: hidden_dim -> emb_dim
            self.fc1 = nn.Linear(emb_dim, hidden_dim, dtype=dtype, bias=False) # W1
            self.fc2 = nn.Linear(emb_dim, hidden_dim, dtype=dtype, bias=False) # W3 (게이트 분기)
            self.fc3 = nn.Linear(hidden_dim, emb_dim, dtype=dtype, bias=False) # W2
            self.forward = self._gated_forward
        else:
            self.fc_in  = nn.Linear(emb_dim, hidden_dim, dtype=dtype, bias=bias)
            self.fc_out = nn.Linear(hidden_dim, emb_dim, dtype=dtype, bias=bias)
            self.layers = nn.Sequential(
                nn.Linear(emb_dim, hidden_dim, dtype=dtype),
                self.activation,
                nn.Linear(hidden_dim, emb_dim, dtype=dtype),
            )
            self.forward = self._sequential_forward

    def _gated_forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = self.activation(x_fc1) * x_fc2
        return self.fc3(x)

    def _sequential_forward(self, x):
        return self.layers(x)


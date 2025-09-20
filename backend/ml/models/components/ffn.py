import torch
import torch.nn as nn

from .activation_function import get_activation

class CustomFFN(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int = 3072, activation: str = "GELU", is_gated: bool = False, dtype=torch.float32, bias: bool = False):
        super().__init__()
        act_key = activation.lower()
        if act_key in {"swiglu", "swi-glu"} and not is_gated:
            raise ValueError("SWiGLU 활성화를 쓰려면 is_gated=True 여야 합니다.")

        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.is_gated = is_gated
        # SWiGLU 문자열이 들어오면 자동 전환
        self.activation = get_activation("silu" if act_key in {"swiglu", "swi-glu"} else act_key)
        
        if self.is_gated:
            # SwiGLU/GLU 계열
            self.fc1 = nn.Linear(emb_dim, hidden_dim, dtype=dtype, bias=bias) # W1 
            self.fc2 = nn.Linear(emb_dim, hidden_dim, dtype=dtype, bias=bias) # W3 (게이트 분기)
            self.fc3 = nn.Linear(hidden_dim, emb_dim, dtype=dtype, bias=bias) # W2
        else:
            # 순차형 MLP
            self.fc_in  = nn.Linear(emb_dim, hidden_dim, dtype=dtype, bias=bias)
            self.fc_out = nn.Linear(hidden_dim, emb_dim, dtype=dtype, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_gated:
            a = self.activation(self.fc1(x))
            b = self.fc2(x)
            h = a * b
            h = self.fc3(h)
            return h
        else:
            h = self.fc_in(x)
            h = self.activation(h)
            h = self.fc_out(h)
            return h

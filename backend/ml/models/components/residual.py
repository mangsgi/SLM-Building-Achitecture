# components/residual.py

import torch
import torch.nn as nn

class ResidualConnection(nn.Module):
    def __init__(self, source_id: str):
        super().__init__()
        self.source_id = source_id  # 참조할 레이어의 ID만 저장

    def forward(self, x):
        # 실제 연결은 CustomSequential에서 처리됨
        return x

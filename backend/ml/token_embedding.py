import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Token 임베딩 레이어 (vocab_size × d_model)"""
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x)

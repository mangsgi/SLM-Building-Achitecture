import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, dtype=torch.float32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, dtype=dtype)

    def forward(self, x):
        return self.embedding(x)

import torch
import torch.nn as nn
import numpy as np


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, ctx_length, emb_dim, dtype=torch.float32):
        super().__init__()
        self.position_embeddings = nn.Embedding(ctx_length, emb_dim, dtype=dtype)

    def forward(self, x):
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return self.position_embeddings(positions)


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, ctx_length, emb_dim, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        self.pe = self._generate_sinusoidal_embeddings(ctx_length, emb_dim)

    def _generate_sinusoidal_embeddings(self, seq_length, d_model):
        position = np.arange(seq_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = np.zeros((seq_length, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return torch.tensor(pe, dtype=self.dtype).unsqueeze(0)

    def forward(self, x):
        return self.pe[:, :x.shape[1], :].to(x.device)


class RelativePositionalEmbedding(nn.Module):
    def __init__(self, ctx_length, emb_dim, dtype=torch.float32):
        super().__init__()
        self.max_len = ctx_length
        self.rel_emb = nn.Embedding(2 * ctx_length, emb_dim, dtype=dtype)

    def forward(self, q, k):
        q_len = q.shape[1]
        k_len = k.shape[1]
        position_ids = torch.arange(q_len, device=q.device).unsqueeze(1) - torch.arange(k_len, device=k.device).unsqueeze(0)
        position_ids = position_ids + self.max_len
        return self.rel_emb(position_ids)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, emb_dim, ctx_length, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        position = torch.arange(ctx_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2) * -(np.log(10000.0) / emb_dim))
        self.theta = torch.zeros(ctx_length, emb_dim, dtype=self.dtype)
        self.theta[:, 0::2] = torch.cos(position * div_term)
        self.theta[:, 1::2] = torch.sin(position * div_term)

    def forward(self, x, pos):
        theta = self.theta[pos].to(x.device)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_rotated = torch.zeros_like(x)
        x_rotated[..., 0::2] = x_even * theta[:, 0::2] - x_odd * theta[:, 1::2]
        x_rotated[..., 1::2] = x_even * theta[:, 1::2] + x_odd * theta[:, 0::2]

        return x_rotated

import torch
import torch.nn as nn
import numpy as np


class LearnedPositionalEmbedding(nn.Module):
    """GPT2-style 학습 가능한 포지셔널 임베딩"""
    def __init__(self, max_len, d_model):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return self.position_embeddings(positions)


class SinusoidalPositionalEmbedding(nn.Module):
    """Transformer-style 사인/코사인 임베딩"""
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = self._generate_sinusoidal_embeddings(max_len, d_model)

    def _generate_sinusoidal_embeddings(self, seq_length, d_model):
        position = np.arange(seq_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pe = np.zeros((seq_length, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        return torch.tensor(pe, dtype=torch.float32).unsqueeze(0)

    def forward(self, x):
        return self.pe[:, :x.shape[1], :]


class RelativePositionalEmbedding(nn.Module):
    """Transformer-XL / T5-style 상대 포지셔널 임베딩"""
    def __init__(self, max_len, d_model):
        super().__init__()
        self.max_len = max_len
        self.rel_emb = nn.Embedding(2 * max_len, d_model)

    def forward(self, q, k):
        q_len = q.shape[1]
        k_len = k.shape[1]
        position_ids = torch.arange(q_len, device=q.device).unsqueeze(1) - torch.arange(k_len, device=k.device).unsqueeze(0)
        position_ids = position_ids + self.max_len
        return self.rel_emb(position_ids)


class RotaryPositionalEmbedding(nn.Module):
    """RoPE (Rotary) 포지셔널 인코딩: LLaMA, GPT-NeoX 등"""
    def __init__(self, d_model, max_len):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        self.theta = torch.zeros(max_len, d_model)
        self.theta[:, 0::2] = torch.cos(position * div_term)
        self.theta[:, 1::2] = torch.sin(position * div_term)

    def forward(self, x, pos):
        theta = self.theta[pos]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_rotated = torch.zeros_like(x)
        x_rotated[..., 0::2] = x_even * theta[:, 0::2] - x_odd * theta[:, 1::2]
        x_rotated[..., 1::2] = x_even * theta[:, 1::2] + x_odd * theta[:, 0::2]

        return x_rotated

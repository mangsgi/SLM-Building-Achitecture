import torch
import torch.nn as nn
import math
import numpy as np

from config import GPT_CONFIG_124M as config
from token_embedding import TokenEmbedding


vocab_size = config["vocab_size"]
emb_dim = config["emb_dim"]
context_length = config["context_length"]
tokens = torch.randint(0, vocab_size, (1, 10))  # (batch, seq_len)


# 1. 고정된 위치 정보를 학습 가능한 벡터로 저장 - (GPT2)
class LearnedPositionalEmbedding(nn.Module): 
    def __init__(self, max_len, d_model):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_len, d_model) # 학습

    def forward(self, x):
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # [1, seq_len]
        return self.position_embeddings(positions)

# 예제
def learned_ex():
    pos_embedding_layer = LearnedPositionalEmbedding(context_length, emb_dim)
    positional_embeddings = pos_embedding_layer(tokens)
    print(positional_embeddings.shape)  # 출력: torch.Size([1, 10, 768]) (batch, seq_len, d_model)


# 2. 토큰 간 상대적 거리를 sin, cos 함수로 표현 
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.d_model = d_model
        self.pe = self._generate_sinusoidal_embeddings(max_len, d_model)
    
    def _generate_sinusoidal_embeddings(self, seq_length, d_model):
        position = np.arange(seq_length)[:, np.newaxis]  # Shape: (seq_length, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))  # Shape: (d_model/2,)
        
        # 학습 X, 미리 생성
        pe = np.zeros((seq_length, d_model))
        pe[:, 0::2] = np.sin(position * div_term)  # Apply sin to even indices
        pe[:, 1::2] = np.cos(position * div_term)  # Apply cos to odd indices
        
        return torch.tensor(pe, dtype=torch.float32).unsqueeze(0)  # Shape: (1, seq_length, d_model)
    
    def forward(self, x):
        return self.pe[:, :x.shape[1], :]

# 예제
def sinusoidal_ex():
    pos_embedding_layer = SinusoidalPositionalEmbedding(context_length, emb_dim)
    positional_embeddings = pos_embedding_layer(tokens)
    print(positional_embeddings.shape)  # Should output: (1, 10, 768)

"""
word embedding + positional embedding 단순 덧셈(두 임베딩 차원 동일)
==========================================================================================================================

"""

# 3. 일반적인 위치 인코딩 대신, 토큰 간 상대적 거리를 학습 - (T5, Transformer-XL)
class RelativePositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # 상대적 위치 행렬을 위한 임베딩 테이블 생성 - 상대적 위치이므로 행 크기 2배
        self.rel_emb = nn.Embedding(2 * max_len, d_model)
    
    def forward(self, q, k):
        q_len = q.shape[1]
        k_len = k.shape[1]
        
        # 상대적 위치 행렬 생성 (i - j)
        position_ids = torch.arange(q_len, device=q.device).unsqueeze(1) - torch.arange(k_len, device=k.device).unsqueeze(0)
        position_ids = position_ids + self.max_len  # 음수 방지 (Offset)
        
        return self.rel_emb(position_ids)  # (q_len, k_len, d_model)

# 예제
def relative_ex():
    queries = torch.randint(0, vocab_size, (1, 10, emb_dim))
    keys = torch.randint(0, vocab_size, (1, 10, emb_dim))  
    
    pos_embedding_layer = RelativePositionalEmbedding(context_length, emb_dim)
    relative_positional_embeddings = pos_embedding_layer(queries, keys)
    
    print(relative_positional_embeddings.shape)  # Should output: (token_len, token_len, emb_dim) -> 각 토큰 사이의 상대적 거리에 대한 임베딩 값


# 4. 위치를 임베딩 벡터 회전(rotate) 연산으로 인코딩 - (LLaMA, GPT-4)
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(1)  # [seq_len, 1]
        sinusoid_inp = positions * self.inv_freq
        sin, cos = sinusoid_inp.sin(), sinusoid_inp.cos()
        emb = torch.cat([sin, cos], dim=-1)  # [seq_len, d_model]

        return emb.unsqueeze(0)  # [1, seq_len, d_model]

# 예제 실행
def rotary_ex():
    rope_embedding = RotaryPositionalEmbedding(emb_dim)
    rope_pos_embeddings = rope_embedding(tokens)
    print(rope_pos_embeddings.shape)  # 출력: torch.Size([1, 10, 512])


if __name__ == "__main__":
    print(tokens.shape)
    learned_ex()
    sinusoidal_ex()
    relative_ex()
    rotary_ex()

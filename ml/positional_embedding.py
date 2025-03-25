import torch
import torch.nn as nn
import math
import numpy as np

from config import GPT_CONFIG_124M as config
from token_embedding import TokenEmbedding

torch.manual_seed(123)
batch_size = config["batch_size"]
vocab_size = config["vocab_size"]
emb_dim = config["emb_dim"]
context_length = config["context_length"]
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokens = torch.randint(0, vocab_size, (batch_size, 10))  # (batch, seq_len)


# 1. 고정된 위치 정보를 학습 가능한 벡터로 저장 - (GPT2)
class LearnedPositionalEmbedding(nn.Module): 
    def __init__(self, max_len, d_model):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_len, d_model) # 학습

    def forward(self, x):
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # [1, seq_len]
        """
        추후에 token embedding에 더해줄 때 모든 배치에 같은 값을 더해준다.
        tok_embeds + pos_embeds -> Broadcasting: [batch, seq_len, emb_dim] + [seq_len, emb_dim]
        """
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
쿼리 & 키 벡터가 input이 되므로 어텐션 계산 중간에 수행됨
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
    queries = torch.randn(1, 10, emb_dim)  # 쿼리 벡터
    keys = torch.randn(1, 10, emb_dim)  # 키 벡터
    
    pos_embedding_layer = RelativePositionalEmbedding(context_length, emb_dim)
    relative_positional_embeddings = pos_embedding_layer(queries, keys) # 어텐션 연산에 쓰이는 토큰 간 상대적 거리 임베딩 벡터
    
    print(relative_positional_embeddings.shape)  # Should output: (token_len, token_len, emb_dim) -> 각 토큰 사이의 상대적 거리에 대한 임베딩 값 (10, 10, 768)


# 4. 위치를 임베딩 벡터 회전(rotate) 연산으로 인코딩 - (LLaMA, GPT-4 등 최신 모델)
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # 위치별 회전 각도 (theta) 생성
        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))  # (d_model/2,)

        # 짝수 인덱스에는 cos, 홀수 인덱스에는 sin 적용
        self.theta = torch.zeros(max_len, d_model)
        self.theta[:, 0::2] = torch.cos(position * div_term)  # 짝수 차원
        self.theta[:, 1::2] = torch.sin(position * div_term)  # 홀수 차원

    def forward(self, x, pos):
        """
        x: (batch_size, seq_len, d_model) - 입력 벡터
        pos: (seq_len,) - 토큰 위치 인덱스
        """
        # 위치 정보에 해당하는 회전 행렬 추출
        theta = self.theta[pos]  # (seq_len, d_model)

        # 짝수 차원: cos(theta) * x_even - sin(theta) * x_odd
        # 홀수 차원: sin(theta) * x_even + cos(theta) * x_odd
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_rotated = torch.zeros_like(x)
        x_rotated[..., 0::2] = x_even * theta[:, 0::2] - x_odd * theta[:, 1::2]
        x_rotated[..., 1::2] = x_even * theta[:, 1::2] + x_odd * theta[:, 0::2]

        return x_rotated

# 예제
def rotary_ex():
    queries = torch.randn(1, 10, emb_dim)  # 쿼리 벡터
    keys = torch.randn(1, 10, emb_dim)  # 키 벡터

    rope = RotaryPositionalEmbedding(emb_dim, context_length)
    position_ids = torch.arange(10)  # (seq_len,)

    queries_rotated = rope(queries, position_ids)
    keys_rotated = rope(keys, position_ids)

    print(queries_rotated.shape)  # (1, 10, 768)
    print(keys_rotated.shape)  # (1, 10, 768)


if __name__ == "__main__":
    print(tokens.shape)
    learned_ex()
    sinusoidal_ex()
    relative_ex()
    rotary_ex()

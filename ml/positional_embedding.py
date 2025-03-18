import torch
import torch.nn as nn
import math

from config import GPT_CONFIG_124M as config


vocab_size = config["vocab_size"]
emb_dim = config["emb_dim"]
context_length = config["context_length"]
tokens = torch.randint(0, vocab_size, (1, 10))  # (batch, seq_len)


# 1. 고정된 위치 정보를 학습 가능한 벡터로 저장 - (GPT2)
class LearnedPositionalEmbedding(nn.Module): 
    def __init__(self, max_len, d_model):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # [1, seq_len]
        return self.position_embeddings(positions)

# 예제
def learned_ex():
    pos_embedding_layer = LearnedPositionalEmbedding(context_length, emb_dim)
    positional_embeddings = pos_embedding_layer(tokens)
    print(positional_embeddings.shape)  # 출력: torch.Size([1, 10, 512]) (batch, seq_len, d_model)


# 2. 일반적인 위치 인코딩 대신, 토큰 간 상대적 거리를 학습 - (T5, Transformer-XL)
class RelativePositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.relative_embeddings = nn.Embedding(2 * max_len, d_model)

    def forward(self, x):
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device).view(-1, 1) - torch.arange(seq_len, device=x.device).view(1, -1)
        positions = positions + (seq_len - 1)  # 음수를 방지
        return self.relative_embeddings(positions)

# 예제
def relative_ex():
    rel_pos_embedding = RelativePositionalEncoding(context_length, emb_dim)
    rel_pos_embeddings = rel_pos_embedding(tokens)
    print(rel_pos_embeddings.shape)  # 출력: torch.Size([10, 10, 512]) 


# 3. 위치를 임베딩 벡터 회전(rotate) 연산으로 인코딩 - (LLaMA, GPT-4)
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
    learned_ex()
    relative_ex()
    rotary_ex()
    print(tokens.shape)
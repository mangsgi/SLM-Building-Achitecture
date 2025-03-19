import torch
import torch.nn as nn

from config import GPT_CONFIG_124M as config


vocab_size = config["vocab_size"]
emb_dim = config["emb_dim"]
tokens = torch.randint(0, vocab_size, (1, 10))  # 랜덤 토큰 10개


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x)

def token_ex():
    embedding_layer = TokenEmbedding(vocab_size, emb_dim)
    embedded_tokens = embedding_layer(tokens)
    print(embedded_tokens.shape)  # 출력: torch.Size([1, 10, 768])


if __name__ == "__main__":
    token_ex()

"""
Tokenizer:
    - 보통 학습하는 데이터셋에 따라 토크나이저가 결정되고, 해당 데이터셋으로 토크나이저를 학습함
    - 따라서 모델 별로 토크나이저가 다른 것처럼 보일지라도 ex) tiktoken.get_encoding("gpt2"), 모델과 동일한 데이터셋으로 학습한 토크나이저를 불러오기 위함.
    - DIY 모델 학습에선 하나의 데이터셋만 사용할 예정이니, 데이터셋에 따라 토크나이저가 달라지는게 옳은 방향
    - 추후 학습할 데이터 셋 선정 후, 토크나이저를 학습
    - 사용자 입장에선 학습할 데이터 셋을 고르면 그에 맞는 토크나이저로 토큰화 진행
"""
import torch.nn as nn
from ml.token_embedding import TokenEmbedding
from ml.positional_embedding import (
    LearnedPositionalEmbedding,
    SinusoidalPositionalEmbedding,
    RelativePositionalEmbedding,
    RotaryPositionalEmbedding,
)
from ml.attention import MultiHeadAttentionCombinedQKV as MultiHeadAttention
from ml.normalization import LayerNorm


def get_positional_embedding(name: str, context_length: int, emb_dim: int):
    if name == "learned":
        return LearnedPositionalEmbedding(context_length, emb_dim)
    elif name == "sinusoidal":
        return SinusoidalPositionalEmbedding(context_length, emb_dim)
    elif name == "relative":
        return RelativePositionalEmbedding(context_length, emb_dim)
    elif name == "rotary":
        return RotaryPositionalEmbedding(emb_dim, context_length)
    else:
        raise ValueError(f"Unknown positional embedding type: {name}")


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, context_length, drop_rate):
        super().__init__()
        self.norm1 = LayerNorm(emb_dim)
        self.attn = MultiHeadAttention(emb_dim, emb_dim, num_heads, context_length, dropout=drop_rate)
        self.dropout1 = nn.Dropout(drop_rate)
        self.norm2 = LayerNorm(emb_dim)
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Linear(4 * emb_dim, emb_dim)
        )
        self.dropout2 = nn.Dropout(drop_rate)

    def forward(self, x):
        x = x + self.dropout1(self.attn(self.norm1(x)))
        x = x + self.dropout2(self.ff(self.norm2(x)))
        return x


class GPTModel(nn.Module):
    def __init__(self, config: dict, layer_config: dict):
        super().__init__()

        self.ordered_layers = nn.ModuleList()
        self.emb_dim = config["emb_dim"]
        self.context_length = config["context_length"]
        self.num_heads = config["n_heads"]
        self.drop_rate = config["drop_rate"]

        layer_order = layer_config["layer_order"]

        for layer_name in layer_order:
            layer_value = layer_config.get(layer_name)

            if layer_name == "token_embedding" and layer_value:
                self.ordered_layers.append(TokenEmbedding(config["vocab_size"], config["emb_dim"]))

            elif layer_name == "positional_embedding":
                self.ordered_layers.append(get_positional_embedding(layer_value, self.context_length, self.emb_dim))

            elif layer_name == "dropout" and layer_value:
                self.ordered_layers.append(nn.Dropout(self.drop_rate))

            elif layer_name == "transformer_block":
                repeat = layer_value.get("n_layers", 12)  # default to 12 if not specified
                for _ in range(repeat):
                    self.ordered_layers.append(TransformerBlock(self.emb_dim, self.num_heads, self.context_length, self.drop_rate))

            elif layer_name == "final_layernorm" and layer_value:
                self.ordered_layers.append(LayerNorm(self.emb_dim))

            elif layer_name == "linear_output" and layer_value:
                self.ordered_layers.append(nn.Linear(self.emb_dim, config["vocab_size"]))

    def forward(self, x):
        for layer in self.ordered_layers:
            x = layer(x)
        return x

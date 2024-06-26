from dataclasses import dataclass
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class TransformerConfig:
    embed_dim: int
    window_size: int
    vocab_size: int

    attention_head_size: int
    attention_layer_size: int
    hidden_dropout_prob: float

    inference_mode: bool
    device: torch.device


def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                                 mask: torch.Tensor = None):
    dim_k = key.size(-1)

    scores = torch.bmm(query, key.transpose(2, 1)) / sqrt(dim_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    weights = F.softmax(scores, dim=-1)

    attn_outputs = torch.bmm(weights, value)
    return attn_outputs


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.final_norm = nn.LayerNorm(config.embed_dim)
        self.output_layer = nn.Linear(config.embed_dim, config.vocab_size)

    def forward(self, x):
        encoded_val = self.encoder(x)
        encoded_val = self.final_norm(encoded_val)

        logits = self.output_layer(encoded_val)
        return logits

class TransformerEncoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.attention_layer_size)])

    def forward(self, x):
        x = self.embeddings(x)

        for layer in self.layers:
            x = layer(x)

        return x


class Embeddings(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self._config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embeddings = nn.Embedding(config.window_size, config.embed_dim)
        self.layer_norm = nn.LayerNorm(config.embed_dim, eps=1e-12)
        self.dropout = nn.Dropout()

    def forward(self, input_ids):
        # Create position IDs for input sequence
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=self._config.device).unsqueeze(0)

        # Create token and position embeddings
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        # Combine token and position embeddings
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.embed_dim)
        self.layer_norm_2 = nn.LayerNorm(config.embed_dim)

        self.attention = MultiHeadAttention(window_size=config.window_size, embed_dim=config.embed_dim,
                                            num_heads=config.attention_head_size, mask=config.inference_mode)
        self.feed_forward = FeedForward(hidden_size=config.embed_dim, intermediate_size=config.embed_dim * 4,
                                        hidden_dropout_prob=config.hidden_dropout_prob)

    def forward(self, x):
        # Apply layer normalization and then copy input into query, key, value
        hidden_state = self.layer_norm_1(x)
        # Apply attention with a skip connection
        x = x + self.attention(hidden_state)
        # Apply feed-forward layer with a skip connection
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, window_size: int, embed_dim: int, num_heads: int, mask: bool):
        super().__init__()
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(window_size, embed_dim, head_dim, mask) for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_state):
        x = torch.cat([h(hidden_state) for h in self.heads], dim=-1)
        x = self.output_linear(x)
        return x


class AttentionHead(nn.Module):
    def __init__(self, window_size: int, embed_dim: int, head_dim: int, mask: bool = False):
        super().__init__()
        self.is_masking = mask
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

        attention_mask = torch.tril(torch.ones(window_size, window_size)).unsqueeze(0)
        self.register_buffer("mask", attention_mask)

    def forward(self, hidden_state):
        attn_outputs = scaled_dot_product_attention(
            self.q(hidden_state), self.k(hidden_state), self.v(hidden_state))

        return attn_outputs


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_dropout_prob: float):
        super().__init__()
        self.linear_1 = nn.Linear(hidden_size, intermediate_size)
        self.linear_2 = nn.Linear(intermediate_size, hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x

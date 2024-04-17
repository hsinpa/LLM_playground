from math import sqrt
import torch.nn.functional as F
import tiktoken
import torch
import torch.nn as nn
from torch import Tensor

from VanillaTransformer.model import MultiHeadAttention, TransformerConfig, TransformerEncoderLayer, Embeddings, \
    Transformer, TransformerEncoder
from helper_method import load_data_full_text, get_batch

batch_size = 1 # number of batch size for training

shakespeare_texts = load_data_full_text("./assets/tiny_shakespeare.txt")

enc = tiktoken.get_encoding("cl100k_base")
data = torch.tensor(enc.encode(shakespeare_texts), dtype=torch.long)

config = TransformerConfig(embed_dim=768, attention_head_size=12, attention_layer_size=6,
                           hidden_dropout_prob=0.2, window_size=8, vocab_size=enc.n_vocab,
                           inference_mode=True)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
xb, yb = get_batch(train_data, batch_size, config.window_size)
test_data = data[0:config.window_size]

transformer = Transformer(config)
t_result = transformer(xb)

total_params = sum(p.numel() for p in transformer.parameters())
print(t_result.size())
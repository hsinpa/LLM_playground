from math import sqrt
import torch.nn.functional as F
import tiktoken
import torch
import torch.nn as nn
from torch import Tensor

from VanillaTransformer.model import MultiHeadAttention, TransformerConfig, TransformerEncoderLayer, Embeddings, \
    Transformer, TransformerEncoder
from helper_method import load_data_full_text, get_batch

def generate_text_simple(model: nn.Module, idx: torch.Tensor, max_new_tokens: int, context_size: int):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

batch_size = 1 # number of batch size for training


enc = tiktoken.get_encoding("cl100k_base")
data = torch.tensor(enc.encode("Hello, I am"), dtype=torch.long).unsqueeze(0)

config = TransformerConfig(embed_dim=768, attention_head_size=12, attention_layer_size=6,
                           hidden_dropout_prob=0.2, window_size=8, vocab_size=enc.n_vocab,
                           inference_mode=True)


transformer = Transformer(config)
transformer.eval()

out = generate_text_simple(transformer, data, max_new_tokens=5, context_size=config.window_size)
decoded_text = enc.decode(out.squeeze(0).tolist())
print(decoded_text)
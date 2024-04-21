from math import sqrt
import torch.nn.functional as F
import tiktoken
import torch
import torch.nn as nn
from torch import Tensor

from VanillaTransformer.evalution_mehod import generate_text_simple
from VanillaTransformer.model import MultiHeadAttention, TransformerConfig, TransformerEncoderLayer, Embeddings, \
    Transformer, TransformerEncoder
from helper_method import load_data_full_text, get_batch

batch_size = 1 # number of batch size for training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

enc = tiktoken.get_encoding("cl100k_base")
data = torch.tensor(enc.encode("Hello, I am"), dtype=torch.long).unsqueeze(0).to(device)

config = TransformerConfig(embed_dim=768, attention_head_size=12, attention_layer_size=6,
                           hidden_dropout_prob=0.2, window_size=64, vocab_size=enc.n_vocab,
                           inference_mode=True, device=device)


transformer = Transformer(config)
transformer.to(device)
transformer.eval()

out = generate_text_simple(transformer, data, max_new_tokens=5, context_size=config.window_size)
decoded_text = enc.decode(out.squeeze(0).tolist())
print(decoded_text)
from math import sqrt
import torch.nn.functional as F
import tiktoken
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from VanillaTransformer.model import MultiHeadAttention, TransformerConfig, TransformerEncoderLayer, Embeddings, \
    Transformer, TransformerEncoder
from VanillaTransformer.text_data_loader import TextDataSet
from helper_method import load_data_full_text, get_batch

tokenizer = tiktoken.get_encoding("cl100k_base")

batch_size = 1 # number of batch size for training

text_path = ['./assets/the-verdict.txt']

window_size = 10
text_dataset = TextDataSet(text_path, tokenizer, window_size, 1)
test_dataloader = DataLoader(text_dataset, batch_size=6, shuffle=True)


for index, dataset in enumerate(test_dataloader):
    if index > 2:
        break
    print(dataset[0])
# shakespeare_texts = load_data_full_text(text_path[0])
#
# data = torch.tensor(tokenizer.encode(shakespeare_texts), dtype=torch.long)
#
# config = TransformerConfig(embed_dim=768, attention_head_size=12, attention_layer_size=6,
#                            hidden_dropout_prob=0.2, window_size=8, vocab_size=enc.n_vocab,
#                            inference_mode=True)
#
# n = int(0.9 * len(data))
# train_data = data[:n]
# val_data = data[n:]

# xb, yb = get_batch(train_data, batch_size, config.window_size)
# print(xb.shape)
# test_data = data[0:config.window_size]
#
# transformer = Transformer(config)
# t_result = transformer(xb)
#
# total_params = sum(p.numel() for p in transformer.parameters())
# print(t_result.size())
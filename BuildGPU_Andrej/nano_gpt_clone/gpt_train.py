import tiktoken
import torch
from torch import Tensor

from BuildGPU_Andrej.nano_gpt_clone.BigramTorchModel import BigramTorchModel


def load_data(relative_path):
    text = ""
    with open(relative_path, "r", encoding="utf-8") as f:
        text = f.read()

    return text


def get_batch(token_set: Tensor, batch_size: int):
    """Generate a small batch of data of inputs x and targets y"""
    ix = torch.randint(len(token_set) - batch_size, (batch_size,))
    x = torch.stack([token_set[i : i + block_size] for i in ix])
    y = torch.stack([token_set[i + 1 : i + block_size + 1] for i in ix])
    return x, y


shakespeare_texts = load_data("./assets/tiny_shakespeare.txt")

enc = tiktoken.get_encoding("cl100k_base")
data = torch.tensor(enc.encode(shakespeare_texts), dtype=torch.long)
vocab_size = enc.n_vocab
print(f"vocab_size: {vocab_size}")

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8

xb, yb = get_batch(train_data, block_size)
# m = BigramTorchModel(vocab_size)

print(xb)

# out = m(xb, yb)
# print(out.shape)

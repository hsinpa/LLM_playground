import torch
from torch import Tensor

def load_data_full_text(relative_path):
    text = ""
    with open(relative_path, "r", encoding="utf-8") as f:
        text = f.read()

    return text


def get_batch(token_set: Tensor, batch_size: int, window_size: int):
    """Generate a small batch of data of inputs x and targets y"""
    ix = torch.randint(len(token_set) - batch_size, (batch_size,))
    x = torch.stack([token_set[i : i + window_size] for i in ix])
    y = torch.stack([token_set[i + 1 : i + window_size + 1] for i in ix])
    return x, y

import torch


def read_full_text(path: str):

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def get_batch(dataset: torch.tensor, block_size: int, batch_size: int):
    ix = torch.randint(len(dataset) - block_size, (batch_size,))
    x = torch.stack([dataset[i : i + block_size] for i in ix])
    y = torch.stack([dataset[i + 1 : i + block_size + 1] for i in ix])
    return x, y

import torch
from torch import nn
from torch.utils.data import DataLoader


def read_full_text(path: str):

    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def get_batch(dataset: torch.tensor, block_size: int, batch_size: int):
    ix = torch.randint(len(dataset) - block_size, (batch_size,))
    x = torch.stack([dataset[i : i + block_size] for i in ix])
    y = torch.stack([dataset[i + 1 : i + block_size + 1] for i in ix])
    return x, y

def calc_loss_batch(input_batch: torch.Tensor, target_batch: torch.Tensor, model: nn.Module, device: torch.device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)

    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss

def calc_loss_loader(data_loader: DataLoader, model: nn.Module, device: torch.device, num_batches: int = None):
    total_loss = 0.

    # if num_batches is not set, apply stochastic gradient descent
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches
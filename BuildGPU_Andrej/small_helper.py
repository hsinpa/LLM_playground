import torch

from BuildGPU_Andrej.BigramLanguageModel import BigramLangugaeModel


def read_full_text(path: str):

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def get_batch(dataset: torch.tensor, block_size: int, batch_size: int):
    ix = torch.randint(len(dataset) - block_size, (batch_size,))
    x = torch.stack([dataset[i : i + block_size] for i in ix])
    y = torch.stack([dataset[i + 1 : i + block_size + 1] for i in ix])
    return x, y

def estimate_loss(model: BigramLangugaeModel, train_set: torch.tensor, val_set: torch.tensor ,block_size: int, batch_size: int,
                  eval_iters: int):

    out = {}
    model.eval()

    for split in ['train, val']:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            X, Y = get_batch(val_set if split == 'val' else train_set, block_size, batch_size)
            
            logits, loss = model(X, Y)
            losses[k] = loss.item()

        out[split] = losses.mean()
    model.train()
    return out
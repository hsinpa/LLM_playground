import torch
from tiktoken import Encoding
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from BuildGPU_Andrej.small_helper import calc_loss_loader, calc_loss_batch

def generate_text_simple(model: nn.Module, idx: torch.Tensor, max_new_tokens: int, context_size: int,
                         temperature: float = 1.0, top_k: int = None):
    temperature = max(0.01, min(temperature, 1.0))

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        # Apply Temperature
        logits = logits / temperature
        probas = torch.softmax(logits, dim=-1)

        idx_next = torch.multinomial(probas, num_samples=1)

        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def evaluate_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, eval_iter: int):
    model.eval()

    # Use the whole data loader as evaluation step
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)

    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model: nn.Module, tokenizer: Encoding, device: torch.device, start_context: str, context_size: int):
    model.eval()
    encoded = torch.tensor(tokenizer.encode(start_context)).unsqueeze(0).to(device)

    with torch.no_grad():
        token_ids = generate_text_simple(model=model, idx=encoded, max_new_tokens=50, context_size=context_size,
                                         temperature=1.0, top_k=10)

        decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())
        print(decoded_text.replace("\n", " ")) # Compact print format

    model.train()

def train_model_simple(model: nn.Module, tokenizer: Encoding, train_loader: DataLoader, val_loader: DataLoader, optimizer: Optimizer,
                       device: torch.device, num_epochs: int, eval_freq: int, eval_iter: int, start_context: str, context_size: int):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()

            loss = calc_loss_batch(input_batch, target_batch, model, device)

            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        generate_and_print_sample(model, tokenizer, device, start_context, context_size)

    return train_losses, val_losses, track_tokens_seen

def plot_losses(epochs_seen, tokens_seen: list[float], train_losses: list[float], val_losses: list[float]):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax2 = ax1.twiny() #A
    ax2.plot(tokens_seen, train_losses, alpha=0) #B
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()
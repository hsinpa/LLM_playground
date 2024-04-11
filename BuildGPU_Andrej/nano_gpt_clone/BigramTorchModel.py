import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)


class BigramTorchModel(torch.nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets):
        logits = self.token_embedding_table(idx)

        return logits

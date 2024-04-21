import os
from math import sqrt
import torch.nn.functional as F
import tiktoken
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from BuildGPU_Andrej.small_helper import calc_loss_loader
from VanillaTransformer.evalution_mehod import train_model_simple, plot_losses
from VanillaTransformer.model import MultiHeadAttention, TransformerConfig, TransformerEncoderLayer, Embeddings, \
    Transformer, TransformerEncoder
from VanillaTransformer.text_data_loader import TextDataSet, get_data_loader
from helper_method import load_data_full_text, get_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #A
save_path = "./saves/model_and_optimizer.pth"

tokenizer = tiktoken.get_encoding("cl100k_base")

batch_size = 1 # number of batch size for training

train_text_path = ['./assets/the-verdict.txt']
validation_text_path = ['./assets/the-verdict-validation.txt']

window_size = 256

train_loader = get_data_loader(train_text_path, tokenizer, window_size, stride=8, batch_size=16, drop_last=True)
validation_loader = get_data_loader(validation_text_path, tokenizer, window_size, stride=8, batch_size=16, drop_last=False)

config = TransformerConfig(embed_dim=768, attention_head_size=12, attention_layer_size=12,
                           hidden_dropout_prob=0.1, window_size=window_size, vocab_size=tokenizer.n_vocab,
                           inference_mode=False, device=device)

transformer = Transformer(config)
transformer.to(device)

optimizer: Optimizer = torch.optim.AdamW(transformer.parameters(), lr=0.0004, weight_decay=0.1)

if os.path.exists(save_path):
    checkpoint = torch.load(save_path)
    transformer.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

num_epochs = 10

train_losses, val_losses, tokens_seen = train_model_simple(transformer, tokenizer, train_loader, validation_loader,
                                                           optimizer, device, num_epochs=num_epochs, eval_freq=5, eval_iter=1,
                                                           start_context="Every effort moves you", context_size=config.window_size,)

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

torch.save({
    "model_state_dict": transformer.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    },save_path)
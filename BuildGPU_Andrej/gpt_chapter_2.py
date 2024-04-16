import tiktoken
import torch

from BuildGPU_Andrej.GPTDataset import create_dataloader

with open("../assets/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    print("Total number of character:", len(raw_text))

tokenizer = tiktoken.get_encoding("cl100k_base")

max_length = 4
dataloader = create_dataloader(raw_text, tokenizer, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

out_dim = 256
vocab_size = tokenizer.n_vocab
token_embedding_layer = torch.nn.Embedding(vocab_size, out_dim)
token_embeddings = token_embedding_layer(inputs)

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, out_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)
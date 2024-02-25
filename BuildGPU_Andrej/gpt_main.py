import torch

from BuildGPU_Andrej.character_tokenizer import CharacterTokenizer
from BuildGPU_Andrej.small_helper import read_full_text, get_batch

text_file = "./assets/tiny_shakespeare.txt"
raw_text = read_full_text(text_file)

tokenizer = CharacterTokenizer(raw_text)

data = torch.tensor(tokenizer.encode(raw_text), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8
batch_size = 4

xb, yb = get_batch(train_data, block_size=block_size, batch_size=batch_size)
print(xb)

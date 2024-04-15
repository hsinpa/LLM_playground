import re

from BuildGPU_Andrej.character_tokenizer import CharacterTokenizer

with open("../assets/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    print("Total number of character:", len(raw_text))

custom_tokenizer = CharacterTokenizer(raw_text, ["<|endoftext|>", "<|unk|>"])
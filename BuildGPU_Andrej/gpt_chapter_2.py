import re
import tiktoken

from BuildGPU_Andrej.character_tokenizer import CharacterTokenizer

with open("../assets/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    print("Total number of character:", len(raw_text))

tokenizer = tiktoken.get_encoding("cl100k_base")

encode_ids = tokenizer.encode(raw_text)
enc_sample = encode_ids[50:]

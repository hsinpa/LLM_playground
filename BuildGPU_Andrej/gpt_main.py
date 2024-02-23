from BuildGPU_Andrej.character_tokenizer import CharacterTokenizer
from BuildGPU_Andrej.small_helper import read_full_text

text_file = "./assets/tiny_shakespeare.txt"
raw_text = read_full_text(text_file)

tokenizer = CharacterTokenizer(raw_text)

encode = tokenizer.encode("hello world")
decode = tokenizer.decode(encode)

print(encode)
print(decode)


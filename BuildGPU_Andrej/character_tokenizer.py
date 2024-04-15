import re

class CharacterTokenizer:

    def __init__(self, full_text: str, extension: list[str]):
        self._full_text = full_text
        self._unique_chars = sorted(list(set(full_text)))
        self._unique_chars.extend(extension)

        self.vocab_lens = len(self._unique_chars)

        self.__string_to_integer = {ch: i for i, ch in enumerate(self._unique_chars)}
        self.__integer_to_string = {i: ch for i, ch in enumerate(self._unique_chars)}

    def encode(self, target: str):
        return [self.__string_to_integer[c] for c in target]

    def decode(self, tokens: list[int]):
        text = " ".join([self.__integer_to_string[i] for i in tokens])
        return re.sub(r'\s+([,.?!"()\'])', r'\1', text)
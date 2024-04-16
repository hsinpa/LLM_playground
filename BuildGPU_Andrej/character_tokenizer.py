import re

class CharacterTokenizer:

    def __init__(self, full_text: str, extension: list[str]):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', full_text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        self._unique_chars = sorted(list(set(preprocessed)))
        self._unique_chars.extend(extension)

        self.vocab_lens = len(self._unique_chars)

        self.__string_to_integer = {ch: i for i, ch in enumerate(self._unique_chars)}
        self.__integer_to_string = {i: ch for i, ch in enumerate(self._unique_chars)}

    def encode(self, target: str):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', target)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.__string_to_integer else "<|unk|>" for item in preprocessed]
        ids = [self.__string_to_integer[s] for s in preprocessed]

        return ids

    def decode(self, tokens: list[int]):
        text = " ".join([self.__integer_to_string[i] for i in tokens])
        return re.sub(r'\s+([,.?!"()\'])', r'\1', text)
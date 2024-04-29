import regex

unicode_type = 'utf-8'
pat_str = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class BytePairTokenizer:
    def __init__(self, vocab_size):
        self._encode_table = {}
        self._decoder_table = {}
        self.vocab_size = 0
        self._vocab_size = vocab_size
        self._pat = regex.compile(pat_str)

    def train(self, corpus: str):
        utf_8_bits = 256
        num_merges = self._vocab_size - utf_8_bits
        num_merges = max(0, min(num_merges, num_merges))  # clamp
        self.vocab_size = utf_8_bits + num_merges

        words = self._pat.findall(corpus)
        print(len(words))

        words_utf_8 = list(
            word.encode(unicode_type) for word in words
        )
        utf_encoding = b''
        for word in words_utf_8:
            utf_encoding += word

        # utf_encoding = corpus.encode(unicode_type)
        utf_array = list(utf_encoding)
        ids = list(utf_array)

        for i in range(num_merges):
            pair_freq = self._get_pair_frequncy(ids)

            if len(pair_freq) == 0:
                break

            pair_max = max(pair_freq, key=pair_freq.get)
            idx = utf_8_bits + i

            ids = self._merge(ids, pair_max, idx)
            self._encode_table[pair_max] = idx
            self.vocab_size = idx

        self._decoder_table = self._get_decoder_table(self._encode_table)
        print(f"Original Length {len(utf_encoding)}")
        print(f"vocab_size {self.vocab_size}")

    def encode(self, text):
        # given a string, return list of integers (the tokens)
        tokens = list(text.encode(unicode_type))
        while len(tokens) >= 2:
            stats = self._get_pair_frequncy(tokens)

            pair = min(stats, key=lambda p: self._encode_table.get(p, float("inf")))

            if pair not in self._encode_table:
                break  # nothing else can be merged

            idx = self._encode_table[pair]
            tokens = self._merge(tokens, pair, idx)
        return tokens

    def decode(self, tokens: list[int]) -> str:
        # concat every thing in the lookup table, and transform to bytes type
        bytes = b"".join(self._decoder_table[idx] for idx in tokens)

        # replace error byte with 'unknown'
        text = bytes.decode(unicode_type, errors="replace")

        return text

    def _get_pair_frequncy(self, tokens: list[int]):
        temp_dict = {}

        for pair in zip(tokens, tokens[1:]):
            if pair in temp_dict:
                temp_dict[pair] += 1
            else:
                temp_dict[pair] = 1

        return temp_dict

    def _merge(self, ids: list[int], pair: tuple[int, int], replacement: int):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                newids.append(replacement)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def _get_decoder_table(self, encode_table: dict[tuple, int]):
        # Set the original utf-8 1 byte data as default
        vocab = {idx: bytes([idx]) for idx in range(256)}

        # value of encode_table, start from 256, and gradually grow larger
        # tuple is guarantee to be smaller than current value
        for (p0, p1), idx in encode_table.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        return vocab

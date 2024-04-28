from Tokenizer.BytePairTokenizer import BytePairTokenizer

corpus = ("Byte pair encoding[1][2] (also known as digram coding)[3] is an algorithm, "
          "first described in 1994 by Philip Gage for encoding strings of text into tabular form for use in downstream modeling.[4] "
          "Its modification is notable as the large language model tokenizer with an ability to combine both tokens that encode single characters "
          "(including single digits or single punctuation marks) and those that encode whole words (even the longest compound words).[5][6][7] "
          "This modification, in the first step, assumes all unique characters to be an initial set of 1-character long n-grams (i.e. initial ). "
          "Then, successively the most frequent pair of adjacent characters is merged into a new, 2-character long n-gram and all instances of the pair are"
          " replaced by this new token. This is repeated until a vocabulary of prescribed size is obtained. Note that new words can always be constructed from final "
          "vocabulary tokens and initial-set characters.[8]")

tokenzier = BytePairTokenizer(vocab_size=320)
tokenzier.train(corpus)
encode_token = tokenzier.encode("曾經拿下去年 EVO Japan 亞軍的台灣選手 ZJZ 曾家鎮則止步準決賽，排名第 13。")
print(len(encode_token))
print(tokenzier.decode(encode_token))

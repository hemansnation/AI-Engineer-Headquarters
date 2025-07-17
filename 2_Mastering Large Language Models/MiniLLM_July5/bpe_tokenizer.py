from collections import defaultdict
import re

class BPETokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []
        self.word_freq = defaultdict(int)
        self.inverse_vocab = {}

    def get_stats(self, word_freq):
        pairs = defaultdict(int)
        for word, freq in word_freq.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def merge_vocab(self, pair, word_freq):
        new_word_freq = defaultdict(int)
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in word_freq:
            w_out = p.sub(''.join(pair), word)
            new_word_freq[w_out] = word_freq[word]
        return new_word_freq

    def train(self, corpus):
        for word in corpus:
            word = ' '.join(list(word)) + ' </w>'
            self.word_freq[word] += 1

        for i in range(self.vocab_size - 256):
            pairs = self.get_stats(self.word_freq)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.merges.append(best)
            self.word_freq = self.merge_vocab(best, self.word_freq)

        self.vocab = {'<pad>': 0, '<unk>': 1, '</w>': 2}
        idx = 3
        for char in range(256):
            self.vocab[chr(char)] = idx
            idx += 1
        for merge in self.merges:
            self.vocab[''.join(merge)] = idx
            idx += 1
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text):
        words = text.split()
        tokens = []
        for word in words:
            word = ' '.join(list(word)) + ' </w>'
            while True:
                pairs = self.get_stats({word: 1})
                if not pairs:
                    break
                best = max(pairs, key=pairs.get)
                if best not in self.merges:
                    break
                word = self.merge_vocab(best, {word: 1}).popitem()[0]
            for token in word.split():
                tokens.append(self.vocab.get(token, self.vocab['<unk>']))
        return tokens

    def decode(self, tokens):
        text = ''.join(self.inverse_vocab.get(t, '<unk>') for t in tokens)
        return text.replace('</w>', ' ')

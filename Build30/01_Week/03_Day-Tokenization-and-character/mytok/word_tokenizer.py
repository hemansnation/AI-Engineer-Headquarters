import re

UNK = '<unk>'


class WordTokenizer:
    def __init__(self):
        self.stoi = {}
        self.itos = {}

    def split(self, text):
        return re.findall(r'\w+|[^\w\s]', text.lower())

    def train(self, text):
        words = sorted(set(self.split(text)))
        vocab = [UNK] + words
        self.stoi = {w: i for i, w in enumerate(vocab)}
        self.itos = {i: w for w, i in self.stoi.items()}

    def encode(self, text):
        unk_id = self.stoi[UNK]
        return [self.stoi.get(w, unk_id) for w in self.split(text)]

    def decode(self, ids):
        return ' '.join(self.itos[i] for i in ids)
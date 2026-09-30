class CharTokenizer:
    def __init__(self):
        self.stoi = {}
        self.itos = {}

    def train(self, text):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return ''.join(self.itos[i] for i in ids)
from collections import Counter


class BPETokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

    @staticmethod
    def count_pairs(ids):
        counts = Counter()
        for a, b in zip(ids, ids[1:]):
            counts[(a, b)] += 1
        return counts

    @staticmethod
    def merge_pair(ids, pair, new_id):
        out = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def train(self, text, vocab_size):
        ids = list(text.encode('utf-8'))
        for _ in range(vocab_size - 256):
            counts = self.count_pairs(ids)
            if not counts:
                break
            pair = max(counts, key=counts.get)
            if counts[pair] < 2:
                break
            new_id = 256 + len(self.merges)
            ids = self.merge_pair(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def encode(self, text):
        ids = list(text.encode('utf-8'))
        while len(ids) >= 2:
            counts = self.count_pairs(ids)
            pair = min(counts, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
            ids = self.merge_pair(ids, pair, self.merges[pair])
        return ids

    def decode_bytes(self, ids):
        return b''.join(self.vocab[i] for i in ids)

    def decode(self, ids):
        return self.decode_bytes(ids).decode('utf-8', errors='replace')

    def pieces(self, ids):
        return [self.vocab[i].decode('utf-8', errors='replace') for i in ids]
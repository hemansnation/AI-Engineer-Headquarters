from pathlib import Path

import numpy as np


class VectorStore:
    def __init__(self, embedder):
        self.embedder = embedder
        self.texts = []
        self.matrix = None

    def load(self, path):
        lines = Path(path).read_text(encoding='utf-8').splitlines()
        self.texts = [line.strip() for line in lines if line.strip()]
        self.matrix = self.embedder.embed_many(self.texts)

    def search(self, query, k=3):
        if not self.texts:
            return []
        scores = self.matrix @ self.embedder.embed_query(query)
        k = min(k, len(self.texts))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [{'text': self.texts[i], 'score': round(float(scores[i]), 4)} for i in top]

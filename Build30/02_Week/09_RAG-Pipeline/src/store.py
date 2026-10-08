import json
from pathlib import Path

import numpy as np

class VectorStore:
    def __init__(self, embed_model: str):
        self.embed_model = embed_model
        self.vectors = np.empty((0, 0), dtype=np.float32)
        self.chunks: list[dict] = []

    def add(self, chunks: list[dict], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError('chunks and vectors must have the same length')
        self.chunks = list(chunks)
        self.vectors = vectors

    def search(self, query_vector: np.ndarray, k: int) -> list[dict]:
        if len(self.chunks) == 0:
            return []
        scores = self.vectors @ query_vector
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [{**self.chunks[i], 'score': float(scores[i])} for i in top]

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / 'vectors.npy', self.vectors)
        meta = {'embed_model': self.embed_model, 'chunks': self.chunks}
        (folder / 'chunks.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, folder: Path, embed_model: str) -> 'VectorStore':
        meta = json.loads((folder / 'chunks.json').read_text(encoding='utf-8'))
        built_with = meta['embed_model']
        if built_with != embed_model:
            raise ValueError(
                f'Index was built with {built_with} but you are querying with {embed_model}. Run the index step again.'
            )
        store = cls(embed_model)
        store.vectors = np.load(folder / 'vectors.npy')
        store.chunks = meta['chunks']
        return store
import hashlib
import os
from pathlib import Path
import numpy as np
from .tracker import timed
DEFAULT_MODEL = os.getenv('EMBED_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
CACHE_DIR = Path('.cache')
class Embedder:
    def __init__(self, model_name=DEFAULT_MODEL):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
    def encode(self, texts):
        vectors = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vectors, dtype=np.float32)
class VectorSearcher:
    def __init__(self, docs, embedder=None, cache=True):
        self.docs = docs
        self.embedder = embedder or Embedder()
        self.matrix = self._load_or_embed(cache)
    def _load_or_embed(self, cache):
        texts = [d['text'] for d in self.docs]
        name = getattr(self.embedder, 'model_name', 'custom')
        digest = hashlib.sha256((name + '|'.join(texts)).encode('utf-8')).hexdigest()[:16]
        path = CACHE_DIR / f'embeddings-{digest}.npy'
        if cache and path.exists():
            return np.load(path)
        matrix = self.embedder.encode(texts)
        if cache:
            CACHE_DIR.mkdir(exist_ok=True)
            np.save(path, matrix)
        return matrix
    @timed('vector')
    def search(self, query, top_n=20):
        query_vector = self.embedder.encode([query])[0]
        scores = self.matrix @ query_vector
        ranked = sorted(range(len(self.docs)), key=lambda i: (-scores[i], i))
        return [(self.docs[i]['id'], float(scores[i])) for i in ranked[:top_n]]

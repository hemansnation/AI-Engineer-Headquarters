import numpy as np
from fastembed import TextEmbedding

from app.cache import LRUCache
from app.config import settings


class Embedder:
    def __init__(self):
        self.model = TextEmbedding(model_name=settings.embed_model)
        self.cache = LRUCache(settings.cache_size)

    def embed_many(self, texts):
        vectors = np.array(list(self.model.embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms

    def embed_query(self, text):
        key = text.strip().lower()
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        vector = self.embed_many([text])[0]
        self.cache.put(key, vector)
        return vector

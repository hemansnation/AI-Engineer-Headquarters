import os
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

class Embedder:
    def __init__(self, model_name=None, query_prefix=None):
        self.model_name = model_name or os.getenv('EMBED_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        if query_prefix is None:
            query_prefix = os.getenv('EMBED_QUERY_PREFIX', '')
        self.query_prefix = query_prefix
        self._model = SentenceTransformer(self.model_name)
        self.dim = int(self._model.encode(['probe']).shape[-1])
    def _encode(self, texts):
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=32,
        )
        return np.asarray(vectors, dtype=np.float32)
    def embed_documents(self, texts):
        return self._encode(texts)
    def embed_query(self, text):
        return self._encode([self.query_prefix + text])[0]
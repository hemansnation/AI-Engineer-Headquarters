import numpy as np
from .config import EMBED_MODEL
_model = None
def _load():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL)
    return _model
def embed(texts):
    vecs = _load().encode(texts, normalize_embeddings=True)
    return np.asarray(vecs)
def similarity(a, b):
    va, vb = embed([a, b])
    return float(np.dot(va, vb))

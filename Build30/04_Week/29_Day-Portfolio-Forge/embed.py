from functools import lru_cache
import numpy as np
from config import EMBED_MODEL
@lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)
def encode(texts: list[str]) -> np.ndarray:
    vectors = get_model().encode(texts, normalize_embeddings=True, batch_size=32)
    return np.asarray(vectors, dtype=np.float32)
def role_fit(role: str, texts: list[str]) -> list[float]:
    vectors = encode([role] + texts)
    return (vectors[1:] @ vectors[0]).tolist()

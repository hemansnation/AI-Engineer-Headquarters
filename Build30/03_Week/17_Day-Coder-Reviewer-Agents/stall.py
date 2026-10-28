import numpy as np
from config import EMBED_MODEL
_model = None
def default_embed(text):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL)
    return _model.encode(text, normalize_embeddings=True)
def cosine(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
def normalize(code):
    return '\n'.join(line.rstrip() for line in code.strip().splitlines())
def is_stalled(previous_code, new_code, embed=None, threshold=0.97):
    if normalize(previous_code) == normalize(new_code):
        return True
    if embed is None:
        return False
    return cosine(embed(previous_code), embed(new_code)) >= threshold

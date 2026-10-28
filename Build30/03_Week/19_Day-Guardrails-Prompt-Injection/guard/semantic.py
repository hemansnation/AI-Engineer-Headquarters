import re
import numpy as np
from guard.config import EMBEDDING_MODEL, SEMANTIC_THRESHOLD
_model = None
def default_embed(texts):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model.encode(texts, normalize_embeddings=True)
def load_phrases(path):
    with open(path, encoding='utf-8') as file:
        return [line.strip() for line in file if line.strip()]
def split_chunks(text):
    parts = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [part.strip() for part in parts if len(part.strip()) >= 12]
class SemanticDetector:
    def __init__(self, phrases, embed_fn=default_embed, threshold=SEMANTIC_THRESHOLD):
        self.phrases = phrases
        self.embed_fn = embed_fn
        self.threshold = threshold
        self.matrix = None
    def scan(self, text):
        if self.matrix is None:
            self.matrix = np.asarray(self.embed_fn(self.phrases))
        chunks = split_chunks(text)
        if not chunks:
            return {'hit': False, 'score': 0.0, 'phrase': None, 'chunk': None}
        vectors = np.asarray(self.embed_fn(chunks))
        scores = vectors @ self.matrix.T
        i, j = np.unravel_index(np.argmax(scores), scores.shape)
        score = float(scores[i, j])
        return {'hit': score >= self.threshold, 'score': round(score, 3), 'phrase': self.phrases[j], 'chunk': chunks[i]}

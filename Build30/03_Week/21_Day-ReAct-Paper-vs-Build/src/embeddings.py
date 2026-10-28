import os
import numpy as np
import ollama
from dotenv import load_dotenv
load_dotenv()
NAME = os.getenv('EMBED_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
_model = None
def embed(texts):
    global _model
    texts = list(texts)
    if NAME.startswith('ollama:'):
        response = ollama.embed(model=NAME.split(':', 1)[1], input=texts)
        vectors = np.asarray(response['embeddings'], dtype='float32')
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(NAME)
    return np.asarray(_model.encode(texts, normalize_embeddings=True))
class TitleIndex:
    def __init__(self, titles):
        self.titles = titles
        self.vectors = embed(titles)
    def closest(self, query, k=5):
        scores = self.vectors @ embed([query])[0]
        order = np.argsort(-scores)[:k]
        return [self.titles[i] for i in order]

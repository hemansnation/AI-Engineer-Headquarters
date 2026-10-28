import os
import numpy as np
from dotenv import load_dotenv
load_dotenv()
MODEL_NAME = os.getenv('EMBED_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
_model = None
def embed(texts):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return np.asarray(_model.encode(texts, normalize_embeddings=True))

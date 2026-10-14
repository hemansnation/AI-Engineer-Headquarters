import hashlib
from dataclasses import dataclass
import bm25s
import numpy as np
from sentence_transformers import SentenceTransformer
from rag.chunker import Chunk
from rag.config import CANDIDATES, EMBED_MODEL, INDEX_DIR, TOP_K
from rag.fusion import rrf


@dataclass
class Hit:
    index: int
    chunk: Chunk
    score: float


class HybridRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.texts = [chunk.text for chunk in chunks]
        self.model = SentenceTransformer(EMBED_MODEL)
        self.vectors = self._load_or_embed()
        self.bm25 = bm25s.BM25()
        tokens = bm25s.tokenize(self.texts, stopwords='en', show_progress=False)
        self.bm25.index(tokens, show_progress=False)

    def _load_or_embed(self):
        INDEX_DIR.mkdir(exist_ok=True)
        joined = EMBED_MODEL + '||'.join(self.texts)
        key = hashlib.sha256(joined.encode('utf-8')).hexdigest()[:16]
        path = INDEX_DIR / f'vectors_{key}.npy'
        if path.exists():
            return np.load(path)
        vectors = self.model.encode(self.texts, normalize_embeddings=True, show_progress_bar=False)
        np.save(path, vectors)
        return vectors

    def embed_query(self, query):
        if 'query' in (self.model.prompts or {}):
            return self.model.encode(query, prompt_name='query', normalize_embeddings=True)
        return self.model.encode(query, normalize_embeddings=True)

    def vector_search(self, query, k):
        query_vector = self.embed_query(query)
        scores = self.vectors @ query_vector
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order]

    def keyword_search(self, query, k):
        k = min(k, len(self.texts))
        tokens = bm25s.tokenize(query, stopwords='en', show_progress=False)
        if not tokens.ids or not tokens.ids[0]:
            return []
        ids, scores = self.bm25.retrieve(tokens, k=k, show_progress=False)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if s > 0]

    def search(self, query, mode='hybrid', top_k=TOP_K):
        if mode == 'vector':
            ranked = self.vector_search(query, top_k)
        elif mode == 'keyword':
            ranked = self.keyword_search(query, top_k)
        else:
            vector_list = self.vector_search(query, CANDIDATES)
            keyword_list = self.keyword_search(query, CANDIDATES)
            ranked = rrf([vector_list, keyword_list])[:top_k]
        return [Hit(index=i, chunk=self.chunks[i], score=s) for i, s in ranked]

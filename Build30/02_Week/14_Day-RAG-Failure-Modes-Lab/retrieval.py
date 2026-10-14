import re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from config import EMBED_MODEL, RERANK_MODEL, RRF_K
_models = {}
def get_embedder():
    if 'embed' not in _models:
        _models['embed'] = SentenceTransformer(EMBED_MODEL)
    return _models['embed']
def get_reranker():
    if 'rerank' not in _models:
        _models['rerank'] = CrossEncoder(RERANK_MODEL)
    return _models['rerank']
def tokenize(text):
    return re.findall(r'[a-z0-9]+', text.lower())
class Index:
    def __init__(self, chunks):
        self.chunks = chunks
        texts = [c['text'] for c in chunks]
        self.vectors = get_embedder().encode(texts, normalize_embeddings=True)
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])
    def vector_search(self, query, k, allowed=None):
        q = get_embedder().encode([query], normalize_embeddings=True)[0]
        return self._top(self.vectors @ q, k, allowed)
    def keyword_search(self, query, k, allowed=None):
        scores = np.array(self.bm25.get_scores(tokenize(query)))
        return self._top(scores, k, allowed)
    def _top(self, scores, k, allowed):
        out = []
        for i in np.argsort(-scores):
            if allowed is not None and int(i) not in allowed:
                continue
            out.append((int(i), float(scores[i])))
            if len(out) == k:
                break
        return out
def rrf(rankings, k=RRF_K):
    fused = {}
    for ranking in rankings:
        for rank, (idx, _) in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda pair: pair[1], reverse=True)
def rerank(query, chunks, idxs):
    if not idxs:
        return []
    scores = get_reranker().predict([(query, chunks[i]['text']) for i in idxs])
    ranked = sorted(zip(idxs, scores), key=lambda pair: pair[1], reverse=True)
    return [(int(i), float(s)) for i, s in ranked]

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from .fusion import reciprocal_rank_fusion
from .keyword_search import KeywordSearcher
from .tracker import timed
from .vector_search import VectorSearcher
@dataclass
class Result:
    doc_id: str
    text: str
    source: str
    rrf_score: float
    vector_rank: int
    keyword_rank: int
class HybridSearcher:
    def __init__(self, docs, embedder=None, k=60, pool_size=20, weights=None):
        self.docs = {d['id']: d for d in docs}
        self.keyword = KeywordSearcher(docs)
        self.vector = VectorSearcher(docs, embedder=embedder)
        self.k = k
        self.pool_size = pool_size
        self.weights = weights
    @timed('total')
    def search(self, query, top_n=5):
        with ThreadPoolExecutor(max_workers=2) as pool:
            vector_future = pool.submit(self.vector.search, query, self.pool_size)
            keyword_future = pool.submit(self.keyword.search, query, self.pool_size)
            vector_hits = vector_future.result()
            keyword_hits = keyword_future.result()
        vector_ids = [doc_id for doc_id, _ in vector_hits]
        keyword_ids = [doc_id for doc_id, _ in keyword_hits]
        fused = reciprocal_rank_fusion([vector_ids, keyword_ids], k=self.k, top_n=top_n, weights=self.weights)
        results = []
        for doc_id, score in fused:
            doc = self.docs[doc_id]
            v_rank = vector_ids.index(doc_id) + 1 if doc_id in vector_ids else 0
            k_rank = keyword_ids.index(doc_id) + 1 if doc_id in keyword_ids else 0
            results.append(Result(doc_id, doc['text'], doc['source'], round(score, 5), v_rank, k_rank))
        return results

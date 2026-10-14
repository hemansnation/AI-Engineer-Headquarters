from rank_bm25 import BM25Okapi
from .tokens import tokenize
from .tracker import timed
class KeywordSearcher:
    def __init__(self, docs):
        self.docs = docs
        self.index = BM25Okapi([tokenize(d['text']) for d in docs])
    @timed('keyword')
    def search(self, query, top_n=20):
        scores = self.index.get_scores(tokenize(query))
        ranked = sorted(range(len(self.docs)), key=lambda i: (-scores[i], i))
        return [(self.docs[i]['id'], float(scores[i])) for i in ranked[:top_n] if scores[i] > 0]

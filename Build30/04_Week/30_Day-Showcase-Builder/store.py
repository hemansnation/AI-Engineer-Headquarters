import json
import re
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi
def tokenize(text):
    return [t.strip('.') for t in re.findall(r'[\w.%]+', text.lower())]
def rrf(rankings, k=60):
    fused = {}
    for ranking in rankings:
        for rank, index in enumerate(ranking, start=1):
            fused[index] = fused.get(index, 0) + 1 / (k + rank)
    return fused
class EvidenceStore:
    def __init__(self, embed_fn):
        self.embed_fn = embed_fn
        self.records = []
        self.vectors = np.zeros((0, 0))
        self.bm25 = None
    def add_many(self, chunks):
        self.records = list(chunks)
        self.vectors = np.asarray(self.embed_fn([c['text'] for c in self.records]))
        self.bm25 = BM25Okapi([tokenize(c['text']) for c in self.records])
    def query(self, question, top_k=3, source_contains=None):
        allowed = [i for i, r in enumerate(self.records) if source_contains is None or source_contains in r['source']]
        if not allowed:
            return []
        question_vector = np.asarray(self.embed_fn([question]))[0]
        similarities = self.vectors[allowed] @ question_vector
        vector_rank = [allowed[i] for i in np.argsort(-similarities)]
        keyword_scores = self.bm25.get_scores(tokenize(question))
        keyword_rank = sorted(allowed, key=lambda i: -keyword_scores[i])
        fused = rrf([vector_rank, keyword_rank])
        best = sorted(fused, key=fused.get, reverse=True)[:top_k]
        return [{**self.records[i], 'score': round(fused[i], 5)} for i in best]
    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / 'vectors.npy', self.vectors)
        (folder / 'records.json').write_text(json.dumps(self.records), encoding='utf-8')
    @classmethod
    def load(cls, folder, embed_fn):
        folder = Path(folder)
        store = cls(embed_fn)
        store.records = json.loads((folder / 'records.json').read_text(encoding='utf-8'))
        store.vectors = np.load(folder / 'vectors.npy')
        store.bm25 = BM25Okapi([tokenize(r['text']) for r in store.records])
        return store

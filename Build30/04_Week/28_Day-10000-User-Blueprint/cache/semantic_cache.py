import hashlib
import time
import numpy as np
class HashEmbedder:
    def __init__(self, dim=256):
        self.dim = dim
    def embed(self, text):
        vec = np.zeros(self.dim)
        for word in text.lower().split():
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec
class SentenceEmbedder:
    def __init__(self, model_name='BAAI/bge-small-en-v1.5'):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
    def embed(self, text):
        return self.model.encode(text, normalize_embeddings=True)
class SemanticCache:
    def __init__(self, embedder, threshold=0.92, ttl_seconds=3600, clock=time.time):
        self.embedder = embedder
        self.threshold = threshold
        self.ttl = ttl_seconds
        self.clock = clock
        self.entries = {}
        self.hits = 0
        self.misses = 0
    def get(self, query, tenant, doc_version):
        query_vec = self.embedder.embed(query)
        now = self.clock()
        best_answer, best_score = None, 0.0
        for entry in self.entries.get(tenant, []):
            if entry['doc_version'] != doc_version:
                continue
            if now - entry['created'] > self.ttl:
                continue
            score = float(np.dot(query_vec, entry['vec']))
            if score > best_score:
                best_answer, best_score = entry['answer'], score
        if best_answer is not None and best_score >= self.threshold:
            self.hits += 1
            return best_answer
        self.misses += 1
        return None
    def put(self, query, answer, tenant, doc_version):
        self.entries.setdefault(tenant, []).append({
            'vec': self.embedder.embed(query),
            'answer': answer,
            'doc_version': doc_version,
            'created': self.clock(),
        })
    def invalidate_tenant(self, tenant):
        self.entries.pop(tenant, None)
    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

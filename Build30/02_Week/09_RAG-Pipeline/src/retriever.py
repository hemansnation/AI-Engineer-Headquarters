from src.embedder import Embedder
from src.store import VectorStore

def retrieve(question: str, embedder: Embedder, store: VectorStore, top_k: int) -> list[dict]:
    query_vector = embedder.embed([question])[0]
    return store.search(query_vector, top_k)

def keep_relevant(hits: list[dict], min_score: float) -> list[dict]:
    return [hit for hit in hits if hit['score'] >= min_score]

def select_context(hits: list[dict], budget_words: int) -> list[dict]:
    chosen = []
    used = 0
    for hit in hits:
        size = len(hit['text'].split())
        if used + size <= budget_words:
            chosen.append(hit)
            used += size
    if not chosen and hits:
        chosen = [hits[0]]
    return chosen
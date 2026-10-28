import asyncio
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
class Retriever:
    def __init__(self, model_name: str, corpus_path: str):
        self.model = SentenceTransformer(model_name)
        self.corpus_path = corpus_path
        self.index = self._build_index()
    def _build_index(self):
        lines = Path(self.corpus_path).read_text(encoding='utf-8').splitlines()
        chunks = [line.strip() for line in lines if line.strip()]
        matrix = self.model.encode(chunks, normalize_embeddings=True)
        return chunks, matrix
    def reload(self) -> None:
        self.index = self._build_index()
    @property
    def size(self) -> int:
        return len(self.index[0])
    def _search(self, question: str, top_k: int):
        chunks, matrix = self.index
        query = self.model.encode([question], normalize_embeddings=True)[0]
        scores = matrix @ query
        best = np.argsort(scores)[::-1][:top_k]
        return [(int(i), chunks[i], float(scores[i])) for i in best]
    async def search(self, question: str, top_k: int):
        return await asyncio.to_thread(self._search, question, top_k)

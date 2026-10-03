import json
import os
from pathlib import Path
import numpy as np
from my_vdb.tracker import track

FORMAT_VERSION = 1

class TinyVectorDB:
    def __init__(self, embedder):
        self.embedder = embedder
        self._ids = []
        self._texts = []
        self._meta = []
        self._rows = []
        self._index = {}
        self._matrix = None
    def __len__(self):
        return len(self._ids)
    def _put(self, id, text, vector, metadata):
        if id in self._index:
            i = self._index[id]
            self._texts[i] = text
            self._meta[i] = metadata
            self._rows[i] = vector
        else:
            self._index[id] = len(self._ids)
            self._ids.append(id)
            self._texts.append(text)
            self._meta.append(metadata)
            self._rows.append(vector)
        self._matrix = None
    def _get_matrix(self):
        if self._matrix is None:
            self._matrix = np.vstack(self._rows).astype(np.float32)
        return self._matrix
    @track
    def add_document(self, id, text, metadata=None):
        vector = self.embedder.embed_documents([text])[0]
        self._put(id, text, vector, metadata or {})
    @track
    def add_many(self, documents):
        if not documents:
            return
        texts = [text for _, text, _ in documents]
        vectors = self.embedder.embed_documents(texts)
        for (id, text, metadata), vector in zip(documents, vectors):
            self._put(id, text, vector, metadata or {})
    def get(self, id):
        if id not in self._index:
            return None
        i = self._index[id]
        return {'id': id, 'text': self._texts[i], 'metadata': self._meta[i]}
    @track
    def query(self, search_text, top_k=3, where=None):
        if not self._ids or top_k <= 0:
            return []
        rows = None
        if where:
            rows = [
                i for i, meta in enumerate(self._meta)
                if all(meta.get(key) == value for key, value in where.items())
            ]
            if not rows:
                return []
        query_vec = self.embedder.embed_query(search_text)
        pool = self._get_matrix() if rows is None else self._get_matrix()[rows]
        scores = pool @ query_vec
        best = np.argsort(-scores)[:top_k]
        results = []
        for j in best:
            i = int(j) if rows is None else rows[int(j)]
            results.append({
                'id': self._ids[i],
                'score': float(scores[j]),
                'text': self._texts[i],
                'metadata': self._meta[i],
            })
        return results
    @track
    def save_to_disk(self, filepath):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = {
            'format_version': FORMAT_VERSION,
            'embed_model': self.embedder.model_name,
            'dim': self.embedder.dim,
            'ids': self._ids,
            'texts': self._texts,
            'metadata': self._meta,
        }
        if self._ids:
            vectors = self._get_matrix()
        else:
            vectors = np.zeros((0, self.embedder.dim), dtype=np.float32)
        tmp = path.with_suffix(path.suffix + '.tmp')
        with tmp.open('wb') as f:
            np.savez_compressed(f, vectors=vectors, meta=np.array(json.dumps(header)))
        os.replace(tmp, path)
    @classmethod
    def load_from_disk(cls, filepath, embedder):
        with np.load(Path(filepath), allow_pickle=False) as data:
            vectors = data['vectors']
            header = json.loads(data['meta'].item())
        saved_model = header['embed_model']
        saved_dim = header['dim']
        if saved_model != embedder.model_name or saved_dim != embedder.dim:
            raise ValueError(
                f'File was built with {saved_model} ({saved_dim} dims), '
                f'but you loaded {embedder.model_name} ({embedder.dim} dims). Rebuild the store.'
            )
        db = cls(embedder)
        for i, doc_id in enumerate(header['ids']):
            db._put(doc_id, header['texts'][i], vectors[i], header['metadata'][i])
        return db
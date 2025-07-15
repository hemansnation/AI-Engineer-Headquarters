from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
from rank_bm25 import BM25Okapi
import yaml

class Retriever:
    def __init__(self, config):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Updated model
        self.reranker = SentenceTransformer('BAAI/bge-m3')
        self.index = faiss.read_index(config['retrieval']['index_path'])
        with open(config['retrieval']['chunks_path'], 'rb') as f:
            self.chunks = pickle.load(f)
        with open(config['retrieval']['metadata_path'], 'rb') as f:
            self.metadata = pickle.load(f)
        self.bm25 = BM25Okapi([chunk.split() for chunk in self.chunks])
    
    def hybrid_search(self, query, top_k=10, alpha=0.5):
        query_embedding = self.model.encode([query])[0]
        distances, indices = self.index.search(np.array([query_embedding]), top_k)
        
        bm25_scores = self.bm25.get_scores(query.split())
        bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]
        
        combined_indices = list(set(list(indices[0]) + list(bm25_indices)))
        combined_chunks = [self.chunks[i] for i in combined_indices]
        
        query_embedding = self.reranker.encode([query], normalize_embeddings=True)[0]
        chunk_embeddings = self.reranker.encode(combined_chunks, normalize_embeddings=True)
        scores = np.dot(chunk_embeddings, query_embedding)
        sorted_indices = np.argsort(scores)[::-1][:5]
        top_chunks = [combined_chunks[i] for i in sorted_indices]
        top_metadata = [self.metadata[combined_indices[i]] for i in sorted_indices]
        
        return top_chunks, top_metadata

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    retriever = Retriever(config)
    chunks, metadata = retriever.hybrid_search("How to reset my account password?")
    print(chunks)
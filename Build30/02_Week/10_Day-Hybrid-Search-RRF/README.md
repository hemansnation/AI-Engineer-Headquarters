## hybrid search and reciprocal rank fusion

main.py > vector search (embeddings) + keyword search (BM25) in parallel > RRF merge > top 5
evaluate.py > hit@1, recall@5, MRR for vector, keyword, hybrid

no API key. model: all-MiniLM-L6-v2. switch with EMBED_MODEL=BAAI/bge-m3

## production rag 2.0: hybrid search over support docs

main.py ask / compare / eval > rag (chunker, retriever, fusion, generator, evaluate) > eval_results.md
vector + BM25 fused with RRF, answered by local qwen3:4b via ollama. no API key.

run: python seed_data.py, then python main.py eval
failures go in FAILURES.md

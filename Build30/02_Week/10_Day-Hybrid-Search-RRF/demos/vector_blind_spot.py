import sys
sys.path.insert(0, '.')
from corpus import DOCS
from my_search.vector_search import VectorSearcher
searcher = VectorSearcher(DOCS)
for query in ['Error code 0x80070005', 'Part #AB-9128', 'invoice INV-2026-00431']:
    print(f'query: {query}')
    for doc_id, score in searcher.search(query, top_n=3):
        print(f'   {doc_id}  {score:.3f}')

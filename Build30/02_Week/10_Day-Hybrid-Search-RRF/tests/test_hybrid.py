import sys
sys.path.insert(0, '.')
from corpus import DOCS
from my_search import HybridSearcher
from stub_embedder import StubEmbedder
searcher = HybridSearcher(DOCS, embedder=StubEmbedder())
def test_returns_five_results():
    assert len(searcher.search('Error code 0x80070005', top_n=5)) == 5
def test_exact_error_code_lands_in_top_two():
    ids = [r.doc_id for r in searcher.search('Error code 0x80070005', top_n=5)]
    assert 'd01' in ids[:2]
def test_every_result_came_from_at_least_one_retriever():
    for r in searcher.search('my pod keeps restarting', top_n=5):
        assert r.vector_rank > 0 or r.keyword_rank > 0

from corpus import DOCS
from golden import GOLDEN
from my_search.fusion import reciprocal_rank_fusion
from my_search.keyword_search import KeywordSearcher
from my_search.vector_search import VectorSearcher
def rank_of(expected, ids):
    return ids.index(expected) + 1 if expected in ids else 0
def score_mode(name, rank_lists):
    hits_at_1 = hits_at_5 = 0
    reciprocal = 0.0
    for rank in rank_lists:
        if rank == 1:
            hits_at_1 += 1
        if 1 <= rank <= 5:
            hits_at_5 += 1
        if rank:
            reciprocal += 1.0 / rank
    n = len(rank_lists)
    print(f'{name:<10} hit@1 {hits_at_1}/{n}   recall@5 {hits_at_5}/{n}   MRR {reciprocal / n:.3f}')
def run(embedder=None, k=60):
    vector = VectorSearcher(DOCS, embedder=embedder)
    keyword = KeywordSearcher(DOCS)
    vector_ranks, keyword_ranks, hybrid_ranks = [], [], []
    for query, expected in GOLDEN:
        v_ids = [d for d, _ in vector.search(query, 20)]
        k_ids = [d for d, _ in keyword.search(query, 20)]
        h_ids = [d for d, _ in reciprocal_rank_fusion([v_ids, k_ids], k=k, top_n=20)]
        vector_ranks.append(rank_of(expected, v_ids))
        keyword_ranks.append(rank_of(expected, k_ids))
        hybrid_ranks.append(rank_of(expected, h_ids))
    score_mode('vector', vector_ranks)
    score_mode('keyword', keyword_ranks)
    score_mode('hybrid', hybrid_ranks)
if __name__ == '__main__':
    run()

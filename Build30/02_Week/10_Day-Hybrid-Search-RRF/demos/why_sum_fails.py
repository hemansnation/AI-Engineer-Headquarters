import sys
sys.path.insert(0, '.')
from corpus import DOCS
from my_search.fusion import naive_score_sum, reciprocal_rank_fusion
from my_search.keyword_search import KeywordSearcher
from my_search.vector_search import VectorSearcher
vector = VectorSearcher(DOCS)
keyword = KeywordSearcher(DOCS)
query = sys.argv[1] if len(sys.argv) > 1 else 'my pod keeps restarting'
v_hits = vector.search(query, 20)
k_hits = keyword.search(query, 20)
print(f'query: {query}')
print('vector scores   :', [round(s, 2) for _, s in v_hits[:3]])
print('keyword scores  :', [round(s, 2) for _, s in k_hits[:3]])
print('naive sum top 3 :', [(d, round(s, 2)) for d, s in naive_score_sum([v_hits, k_hits], 3)])
print('rrf top 3       :', [(d, round(s, 4)) for d, s in reciprocal_rank_fusion([[d for d, _ in v_hits], [d for d, _ in k_hits]], top_n=3)])

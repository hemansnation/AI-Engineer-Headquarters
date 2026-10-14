import sys
from corpus import DOCS
from my_search import HybridSearcher
from my_search import tracker
def main():
    query = ' '.join(sys.argv[1:]) or 'Error code 0x80070005'
    searcher = HybridSearcher(DOCS)
    results = searcher.search(query, top_n=5)
    print(f'query: {query}')
    print('-' * 70)
    for place, r in enumerate(results, start=1):
        print(f'{place}. {r.doc_id}  rrf={r.rrf_score}  vector_rank={r.vector_rank}  keyword_rank={r.keyword_rank}')
        print(f'   {r.text}')
    print('-' * 70)
    tracker.report()
if __name__ == '__main__':
    main()

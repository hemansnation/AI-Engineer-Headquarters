import sys
sys.path.insert(0, '.')
from corpus import DOCS
from my_search.keyword_search import KeywordSearcher
searcher = KeywordSearcher(DOCS)
for query in ['I cannot remember my passcode, how do I get back in', 'my pod keeps restarting', 'how many days off do I get each year']:
    print(f'query: {query}')
    hits = searcher.search(query, top_n=3)
    if not hits:
        print('   no keyword match at all')
    for doc_id, score in hits:
        print(f'   {doc_id}  {score:.3f}')

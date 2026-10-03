import sys
from pathlib import Path
from my_vdb.embedder import Embedder
from my_vdb.vector_store import TinyVectorDB

DB_PATH = Path('data') / 'tiny.npz'

DOCS = [
    ('pet-01', 'A kitten is playing with a small toy mouse on the sofa.', {'topic': 'pets', 'source': 'blog'}),
    ('pet-02', 'My dog loves chasing a ball in the park every morning.', {'topic': 'pets', 'source': 'blog'}),
    ('pet-03', 'Adopting a rescue cat takes patience and a quiet room.', {'topic': 'pets', 'source': 'guide'}),
    ('fin-01', 'Index funds are a low cost way to invest for the long term.', {'topic': 'finance', 'source': 'guide'}),
    ('fin-02', 'Setting up an emergency fund protects you from surprise bills.', {'topic': 'finance', 'source': 'blog'}),
    ('code-01', 'Use a virtual environment so each Python project keeps its own libraries.', {'topic': 'code', 'source': 'guide'}),
    ('code-02', 'A hash map gives you fast lookups by key in constant time.', {'topic': 'code', 'source': 'blog'}),
    ('food-01', 'Slow cooked lentil soup with cumin and lemon is easy to make.', {'topic': 'food', 'source': 'blog'}),
]
def build(embedder):
    db = TinyVectorDB(embedder)
    db.add_many(DOCS)
    db.save_to_disk(DB_PATH)
    return db
def show(db, question, top_k=2, where=None):
    print(f'\nQuery: {question}  (where={where})')
    for hit in db.query(question, top_k=top_k, where=where):
        score, doc_id, text = hit['score'], hit['id'], hit['text']
        print(f'  {score:.3f}  {doc_id:<8} {text}')
def main():
    embedder = Embedder()
    if DB_PATH.exists() and '--rebuild' not in sys.argv:
        db = TinyVectorDB.load_from_disk(DB_PATH, embedder)
        print(f'Loaded {len(db)} documents from disk. No embedding needed.')
    else:
        db = build(embedder)
        print(f'Embedded and saved {len(db)} documents.')
    show(db, 'a kitten with a toy')
    show(db, 'how do I protect my savings')
    show(db, 'keeping libraries separate', where={'topic': 'code'})
    show(db, 'something warm to eat', top_k=1, where={'source': 'blog'})
if __name__ == '__main__':
    main()
import re
def split_sentences(text):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p]
def make_chunk(doc, n, text):
    return {'chunk_id': doc['doc_id'] + '#' + str(n), 'doc_id': doc['doc_id'], 'n': n, 'plan': doc['plan'], 'current': doc['current'], 'text': text}
def chunk_fixed(doc, size):
    text = doc['text']
    return [make_chunk(doc, n, text[start:start + size]) for n, start in enumerate(range(0, len(text), size))]
def chunk_sentences(doc, max_words, overlap):
    chunks, current = [], []
    for sentence in split_sentences(doc['text']):
        words = sum(len(s.split()) for s in current) + len(sentence.split())
        if current and words > max_words:
            chunks.append(make_chunk(doc, len(chunks), ' '.join(current)))
            current = current[-overlap:] if overlap else []
        current.append(sentence)
    if current:
        chunks.append(make_chunk(doc, len(chunks), ' '.join(current)))
    return chunks
def merge_unique(texts):
    seen, out = set(), []
    for text in texts:
        for sentence in split_sentences(text):
            if sentence not in seen:
                seen.add(sentence)
                out.append(sentence)
    return ' '.join(out)
def expand(chunks, idx, window=1):
    base = chunks[idx]
    near = [c for c in chunks if c['doc_id'] == base['doc_id'] and abs(c['n'] - base['n']) <= window]
    near.sort(key=lambda c: c['n'])
    return merge_unique([c['text'] for c in near])

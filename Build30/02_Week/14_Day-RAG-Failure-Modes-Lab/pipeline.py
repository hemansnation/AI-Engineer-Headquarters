from config import NAIVE_CHARS, MAX_WORDS, OVERLAP
from chunker import chunk_fixed, chunk_sentences, expand
from retrieval import Index, rrf, rerank
from fixes import detect_plan, allowed_chunks, edge_first, calibrate
NAIVE = {'sentence_chunks': False, 'window': False, 'metadata_filter': False, 'hybrid': False, 'rerank': False, 'reorder': False, 'gate': False}
FIXED = {name: True for name in NAIVE}
class Pipeline:
    def __init__(self, docs, flags):
        self.flags = flags
        self.threshold = None
        self.high = None
        if flags['sentence_chunks']:
            chunks = [c for d in docs for c in chunk_sentences(d, MAX_WORDS, OVERLAP)]
        else:
            chunks = [c for d in docs for c in chunk_fixed(d, NAIVE_CHARS)]
        self.index = Index(chunks)
    def retrieve(self, question, k):
        f = self.flags
        chunks = self.index.chunks
        allowed = allowed_chunks(chunks, detect_plan(question)) if f['metadata_filter'] else None
        ranked = self.index.vector_search(question, 20, allowed)
        if f['hybrid']:
            keyword = self.index.keyword_search(question, 20, allowed)
            ranked = rrf([ranked, keyword])[:10]
        if f['rerank']:
            ranked = rerank(question, chunks, [i for i, _ in ranked])
        return ranked[:k]
    def build_units(self, hits):
        units, seen = [], set()
        for i, score in hits:
            c = self.index.chunks[i]
            text = expand(self.index.chunks, i, 1) if self.flags['window'] else c['text']
            if text in seen:
                continue
            seen.add(text)
            units.append({'doc_id': c['doc_id'], 'chunk_id': c['chunk_id'], 'score': float(score), 'unit': text})
        for n, unit in enumerate(units):
            unit['rank'] = n
        return edge_first(units) if self.flags['reorder'] else units
    def fit_gate(self, golden, k):
        inside, outside = [], []
        for item in golden:
            hits = self.retrieve(item['question'], k)
            score = hits[0][1] if hits else -1e9
            (outside if item['gold_doc'] is None else inside).append(score)
        self.threshold, acc = calibrate(inside, outside)
        self.high = max(sorted(inside)[len(inside) // 2], self.threshold)
        return acc
    def run(self, item, k):
        hits = self.retrieve(item['question'], k)
        top = hits[0][1] if hits else -1e9
        answered = top >= self.threshold if self.flags['gate'] else True
        return {'qid': item['qid'], 'kind': item['kind'], 'question': item['question'], 'gold_doc': item['gold_doc'], 'spans': item['spans'], 'key': item['key'], 'answered': bool(answered), 'top_score': float(top), 'retrieved': self.build_units(hits)}

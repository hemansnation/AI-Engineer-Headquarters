from config import NAIVE_CHARS, MAX_WORDS, OVERLAP
from chunker import chunk_fixed, chunk_sentences
def pct(n, d):
    return 0.0 if d == 0 else 100.0 * n / d
def false_answer_rate(rows):
    miss = [r for r in rows if r['gold_doc'] is None]
    return pct(sum(r['answered'] for r in miss), len(miss))
def false_refusal_rate(rows):
    inside = [r for r in rows if r['gold_doc'] is not None]
    return pct(sum(not r['answered'] for r in inside), len(inside))
def top2_on_edges(rows):
    scored = [r for r in rows if len(r['retrieved']) >= 3]
    good = 0
    for r in scored:
        last = len(r['retrieved']) - 1
        edge_ranks = {u['rank'] for p, u in enumerate(r['retrieved']) if p in (0, last)}
        good += edge_ranks == {0, 1}
    return pct(good, len(scored))
def top1_hit(rows, kind):
    sel = [r for r in rows if r['kind'] == kind]
    good = sum(1 for r in sel if r['retrieved'] and r['retrieved'][0]['doc_id'] == r['gold_doc'])
    return pct(good, len(sel))
def boundary_intact(docs, golden, sentence_mode):
    by_id = {d['doc_id']: d for d in docs}
    sel = [g for g in golden if g['kind'] == 'boundary']
    good = 0
    for g in sel:
        doc = by_id[g['gold_doc']]
        chunks = chunk_sentences(doc, MAX_WORDS, OVERLAP) if sentence_mode else chunk_fixed(doc, NAIVE_CHARS)
        good += any(all(s in c['text'] for s in g['spans']) for c in chunks)
    return pct(good, len(sel))
def print_table(docs, golden, naive_rows, fixed_rows):
    lines = [
        ('Confident answer, no evidence', 'false answers on missing questions', false_answer_rate(naive_rows), false_answer_rate(fixed_rows)),
        ('Lost in the middle', 'best two chunks on the edges', top2_on_edges(naive_rows), top2_on_edges(fixed_rows)),
        ('Chunk boundary loss', 'answer inside one chunk', boundary_intact(docs, golden, False), boundary_intact(docs, golden, True)),
        ('Retrieval drift', 'right document at rank 1', top1_hit(naive_rows, 'drift'), top1_hit(fixed_rows, 'drift')),
    ]
    print('failure mode'.ljust(32) + 'what we measure'.ljust(38) + 'naive'.rjust(8) + 'fixed'.rjust(8))
    for name, what, a, b in lines:
        print(f'{name:32}{what:38}{a:7.0f}%{b:7.0f}%')
    print(f'cost of the gate: false refusals on real questions {false_refusal_rate(naive_rows):.0f}% naive, {false_refusal_rate(fixed_rows):.0f}% fixed')

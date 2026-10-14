from pathlib import Path
from rag.golden_set import GOLDEN

MODES = ['vector', 'keyword', 'hybrid']


def first_hit_rank(hits, must_contain):
    for rank, hit in enumerate(hits, start=1):
        if must_contain.lower() in hit.chunk.text.lower():
            return rank
    return None


def run_eval(retriever, top_k=3):
    report = {}
    for mode in MODES:
        ranks = []
        for case in GOLDEN:
            hits = retriever.search(case['query'], mode=mode, top_k=top_k)
            ranks.append(first_hit_rank(hits, case['must_contain']))
        total = len(ranks)
        report[mode] = {
            'hit_rate': sum(1 for r in ranks if r) / total,
            'mrr': sum(1 / r for r in ranks if r) / total,
            'misses': [GOLDEN[i]['query'] for i, r in enumerate(ranks) if r is None],
        }
    return report


def print_report(report):
    print('mode       hit_rate   mrr')
    for mode, row in report.items():
        hit_rate = row['hit_rate']
        mrr = row['mrr']
        print(f'{mode:<10} {hit_rate:<10.2f} {mrr:.2f}')
        for query in row['misses']:
            print(f'    missed: {query}')


def save_report(report, path='eval_results.md'):
    lines = ['# Retrieval results', '', f'Golden set size: {len(GOLDEN)} questions', '', '| mode | hit rate | MRR |', '| --- | --- | --- |']
    for mode, row in report.items():
        hit_rate = row['hit_rate']
        mrr = row['mrr']
        lines.append(f'| {mode} | {hit_rate:.2f} | {mrr:.2f} |')
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')

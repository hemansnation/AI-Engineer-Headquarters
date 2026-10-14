import argparse
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from chunkers import get_chunker
from chunkers.tokens import get_counter
from budget.knapsack import greedy_by_score, greedy_by_density, knapsack_select, stretch
from demos.embed_and_search import query_prefix
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--file', default='data/sample_policy.md')
    p.add_argument('--model', default='BAAI/bge-small-en-v1.5')
    p.add_argument('--question', default='What happens to my money and my access if I want out of an annual plan?')
    p.add_argument('--budget', type=int, default=220)
    p.add_argument('--candidates', type=int, default=8)
    args = p.parse_args()
    model = SentenceTransformer(args.model)
    counter = get_counter(args.model)
    text = Path(args.file).read_text(encoding='utf-8')
    chunks = get_chunker('recursive', counter, 100, 0).chunk(text, Path(args.file).name)
    vectors = model.encode([c.text for c in chunks], normalize_embeddings=True)
    q = model.encode([query_prefix(args.model) + args.question], normalize_embeddings=True)[0]
    sims = vectors @ q
    top = np.argsort(-sims)[:args.candidates]
    cands = [chunks[i] for i in top]
    values = stretch([float(sims[i]) for i in top])
    weights = [c.token_count for c in cands]
    print(f'question: {args.question}')
    print(f'budget: {args.budget} tokens, candidates: {len(cands)}')
    for i, (c, v) in enumerate(zip(cands, values)):
        print(f'  [{i}] tokens={c.token_count:3} value={v:.2f} heading={c.heading!r}')
    for label, pick in [('best score first', greedy_by_score), ('best value per token', greedy_by_density), ('knapsack dp', knapsack_select)]:
        ids = pick(weights, values, args.budget)
        print(f'{label:22} picks={ids} tokens={sum(weights[i] for i in ids)} value={sum(values[i] for i in ids):.2f}')
if __name__ == '__main__':
    main()

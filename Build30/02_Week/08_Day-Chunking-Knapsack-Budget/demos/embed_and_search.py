import argparse
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from chunkers import get_chunker, STRATEGIES
from chunkers.tokens import get_counter
GOLDEN = [
    ('How many days do I have to ask for a refund on a monthly plan?', 'within 30 days of your first payment'),
    ('What is the refund window for annual plans?', 'within 14 days of payment'),
    ('What uptime does Acme Cloud promise?', '99.9% uptime'),
    ('What does error code 0x80070005 mean?', 'access is denied'),
    ('Can I reach support on a Saturday?', 'open from 10:00 to 16:00 IST'),
    ('How long does account deletion take?', 'from our servers within 30 days'),
]
def query_prefix(model_name):
    if 'bge' in model_name and 'm3' not in model_name:
        return 'Represent this sentence for searching relevant passages: '
    return ''
def squash(text):
    return ' '.join(text.lower().split())
def hit_rate(model, prefix, chunks, k):
    vectors = model.encode([c.text for c in chunks], normalize_embeddings=True)
    hits = 0
    for question, phrase in GOLDEN:
        q = model.encode([prefix + question], normalize_embeddings=True)[0]
        top = np.argsort(-(vectors @ q))[:k]
        if any(squash(phrase) in squash(chunks[i].text) for i in top):
            hits += 1
    return hits / len(GOLDEN)
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--file', default='data/sample_policy.md')
    p.add_argument('--model', default='BAAI/bge-small-en-v1.5')
    p.add_argument('--k', type=int, default=2)
    p.add_argument('--sizes', type=int, nargs='+', default=[48, 96, 192])
    args = p.parse_args()
    model = SentenceTransformer(args.model)
    counter = get_counter(args.model)
    prefix = query_prefix(args.model)
    text = Path(args.file).read_text(encoding='utf-8')
    print(f'model={args.model} max_seq_length={model.max_seq_length} k={args.k}')
    for name in STRATEGIES:
        for size in args.sizes:
            chunks = get_chunker(name, counter, size, size // 5).chunk(text, Path(args.file).name)
            avg = sum(c.token_count for c in chunks) / len(chunks)
            score = hit_rate(model, prefix, chunks, args.k)
            print(f'{name:10} size={size:4} chunks={len(chunks):3} avg_tokens={avg:6.1f} hit@{args.k}={score:.2f}')
if __name__ == '__main__':
    main()

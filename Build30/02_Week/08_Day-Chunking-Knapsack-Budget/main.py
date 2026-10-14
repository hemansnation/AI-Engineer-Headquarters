import argparse
from pathlib import Path
from chunkers import get_chunker, STRATEGIES
from chunkers.tokens import get_counter
from chunkers.quality import report
def parse_args():
    p = argparse.ArgumentParser(description='Day 8 chunking module')
    p.add_argument('files', nargs='*', default=['data/sample_policy.md'])
    p.add_argument('--strategy', choices=list(STRATEGIES), default='recursive')
    p.add_argument('--size', type=int, default=120)
    p.add_argument('--overlap', type=int, default=20)
    p.add_argument('--counter', default='BAAI/bge-small-en-v1.5')
    p.add_argument('--limit', type=int, default=510)
    return p.parse_args()
def main():
    args = parse_args()
    counter = get_counter(args.counter)
    chunker = get_chunker(args.strategy, counter, args.size, args.overlap)
    for path in args.files:
        text = Path(path).read_text(encoding='utf-8')
        chunks = chunker.chunk(text, source=Path(path).name)
        print(f'{path}: {len(chunks)} chunks, strategy={args.strategy}, size={args.size}, overlap={args.overlap}')
        for c in chunks:
            preview = c.text[:60].replace('\n', ' ')
            print(f'  {c.chunk_id:28} tok={c.token_count:4} chars={c.start_char}-{c.end_char} heading={c.heading!r} | {preview}')
        print('  quality:', report(chunks, text, args.limit))
if __name__ == '__main__':
    main()

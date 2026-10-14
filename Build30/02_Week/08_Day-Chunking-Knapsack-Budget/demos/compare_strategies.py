import argparse
from pathlib import Path
from chunkers import get_chunker, STRATEGIES
from chunkers.tokens import get_counter
from chunkers.quality import report
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--file', default='data/sample_policy.md')
    p.add_argument('--size', type=int, default=120)
    p.add_argument('--overlap', type=int, default=20)
    p.add_argument('--counter', default='BAAI/bge-small-en-v1.5')
    args = p.parse_args()
    counter = get_counter(args.counter)
    text = Path(args.file).read_text(encoding='utf-8')
    for name in STRATEGIES:
        chunks = get_chunker(name, counter, args.size, args.overlap).chunk(text, Path(args.file).name)
        print(f'{name:10}', report(chunks, text))
if __name__ == '__main__':
    main()

import argparse
from rag.chunker import load_chunks
from rag.config import TOP_K
from rag.evaluate import print_report, run_eval, save_report
from rag.generator import answer
from rag.retriever import HybridRetriever


def build_retriever():
    chunks = load_chunks()
    if not chunks:
        raise SystemExit('No documents found. Run: python seed_data.py')
    return HybridRetriever(chunks)


def show_hits(label, hits):
    print(f'== {label} ==')
    for number, hit in enumerate(hits, start=1):
        preview = hit.chunk.text[:90].replace('\n', ' ')
        print(f'{number}. {hit.chunk.chunk_id}  score={hit.score:.4f}  {preview}')


def main():
    parser = argparse.ArgumentParser(description='Day 13 production RAG')
    sub = parser.add_subparsers(dest='command', required=True)
    ask_cmd = sub.add_parser('ask', help='answer a question with sources')
    ask_cmd.add_argument('question')
    ask_cmd.add_argument('--mode', default='hybrid', choices=['vector', 'keyword', 'hybrid'])
    compare_cmd = sub.add_parser('compare', help='show top hits for all three modes')
    compare_cmd.add_argument('query')
    sub.add_parser('eval', help='run the golden set')
    args = parser.parse_args()
    retriever = build_retriever()
    if args.command == 'ask':
        result = answer(retriever, args.question, mode=args.mode)
        print(result['answer'])
        print('Sources:')
        for hit in result['hits']:
            print(f'  {hit.chunk.chunk_id}')
    elif args.command == 'compare':
        for mode in ['vector', 'keyword', 'hybrid']:
            show_hits(mode, retriever.search(args.query, mode=mode, top_k=TOP_K))
    elif args.command == 'eval':
        report = run_eval(retriever, top_k=TOP_K)
        print_report(report)
        save_report(report)


if __name__ == '__main__':
    main()

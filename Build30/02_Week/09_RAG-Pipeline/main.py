import argparse
from functools import partial

from src import config
from src.embedder import Embedder
from src.generate import ask_llm
from src.pipeline import RagPipeline

def build_pipeline() -> RagPipeline:
    embedder = Embedder(config.EMBED_MODEL)
    llm = partial(ask_llm, model=config.LLM_MODEL)
    return RagPipeline(embedder, llm)

def show(result: dict) -> None:
    print()
    print(result['answer'])
    print()
    for item in result['sources']:
        number = item['number']
        source = item['source']
        score = item['score']
        mark = 'cited' if item['cited'] else 'unused'
        print(f'  [{number}] {source} | score {score} | {mark}')
    best = result['best_score']
    grounded = result['grounded']
    used = result['used_llm']
    print(f'  best score: {best} | grounded: {grounded} | llm called: {used}')
    print()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['index', 'ask', 'chat'])
    parser.add_argument('question', nargs='?')
    args = parser.parse_args()

    pipeline = build_pipeline()

    if args.command == 'index':
        count = pipeline.index(config.DOCS_DIR, config.INDEX_DIR)
        print(f'Indexed {count} chunks into {config.INDEX_DIR}')
        return

    pipeline.load(config.INDEX_DIR)

    if args.command == 'ask':
        if not args.question:
            parser.error('ask needs a question')
        show(pipeline.ask(args.question))
        return

    while True:
        question = input('You: ').strip()
        if question.lower() in {'exit', 'quit', ''}:
            break
        show(pipeline.ask(question))

if __name__ == '__main__':
    main()
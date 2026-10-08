from pathlib import Path
from typing import Callable

from src import config
from src.embedder import Embedder
from src.evaluate import check_grounding
from src.ingest import build_chunks
from src.prompt import build_messages
from src.retriever import keep_relevant, retrieve, select_context
from src.store import VectorStore

class RagPipeline:
    def __init__(self, embedder: Embedder, llm: Callable[[list[dict]], str]):
        self.embedder = embedder
        self.llm = llm
        self.store: VectorStore | None = None

    def index(self, docs_dir: Path, index_dir: Path) -> int:
        chunks = build_chunks(docs_dir, config.CHUNK_WORDS, config.OVERLAP_WORDS)
        if not chunks:
            raise ValueError(f'No documents found in {docs_dir}')
        vectors = self.embedder.embed([chunk['text'] for chunk in chunks])
        store = VectorStore(self.embedder.model_name)
        store.add(chunks, vectors)
        store.save(index_dir)
        self.store = store
        return len(chunks)

    def load(self, index_dir: Path) -> None:
        self.store = VectorStore.load(index_dir, self.embedder.model_name)

    def ask(self, question: str) -> dict:
        if self.store is None:
            raise RuntimeError('Run index or load before you ask anything')

        hits = retrieve(question, self.embedder, self.store, config.TOP_K)
        best_score = round(hits[0]['score'], 3) if hits else 0.0
        relevant = keep_relevant(hits, config.MIN_SCORE)

        if not relevant:
            return {
                'answer': config.NO_ANSWER,
                'sources': [],
                'grounded': None,
                'used_llm': False,
                'best_score': best_score,
            }

        context = select_context(relevant, config.CONTEXT_BUDGET_WORDS)
        answer = self.llm(build_messages(question, context))

        if answer.strip() == config.NO_ANSWER:
            return {
                'answer': answer,
                'sources': [],
                'grounded': None,
                'used_llm': True,
                'best_score': best_score,
            }

        report = check_grounding(answer, context)
        sources = [
            {
                'number': number,
                'source': hit['source'],
                'chunk_id': hit['id'],
                'score': round(hit['score'], 3),
                'cited': number in report['cited'],
            }
            for number, hit in enumerate(context, start=1)
        ]
        return {
            'answer': answer,
            'sources': sources,
            'grounded': report['grounded'],
            'overlap': report['overlap'],
            'used_llm': True,
            'best_score': best_score,
        }
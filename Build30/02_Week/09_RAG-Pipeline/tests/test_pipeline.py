import re

import numpy as np
import pytest

from src import config
from src.evaluate import check_grounding
from src.ingest import chunk_text
from src.pipeline import RagPipeline
from src.store import VectorStore

class FakeEmbedder:
    model_name = 'fake-bag-of-words'

    def embed(self, texts):
        vectors = np.zeros((len(texts), 256), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r'[a-z0-9]+', text.lower()):
                if len(word) > 3:
                    vectors[row, hash(word) % 256] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

def fake_llm_factory(reply):
    calls = []

    def fake_llm(messages):
        calls.append(messages)
        return reply

    fake_llm.calls = calls
    return fake_llm

def test_chunks_respect_budget_and_overlap():
    sentence = 'This is a sentence with exactly eight words.'
    text = ' '.join([sentence] * 20)
    chunks = chunk_text(text, 'demo.txt', max_words=30, overlap_words=10)
    assert len(chunks) > 1
    assert all(len(c['text'].split()) <= 30 for c in chunks)
    assert chunks[1]['text'].startswith(sentence)

def test_store_returns_best_match_first(tmp_path):
    embedder = FakeEmbedder()
    chunks = [
        {'id': 'a#0', 'source': 'a', 'position': 0, 'text': 'refund within seven days of purchase'},
        {'id': 'b#0', 'source': 'b', 'position': 0, 'text': 'support answers email on weekdays'},
    ]
    store = VectorStore(embedder.model_name)
    store.add(chunks, embedder.embed([c['text'] for c in chunks]))
    hits = store.search(embedder.embed(['how do refund purchase work'])[0], 2)
    assert hits[0]['id'] == 'a#0'
    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path, embedder.model_name)
    assert len(loaded.chunks) == 2
    with pytest.raises(ValueError):
        VectorStore.load(tmp_path, 'another-model')

def test_pipeline_answers_with_sources(tmp_path):
    llm = fake_llm_factory('You get a full refund within 7 days of purchase [1].')
    pipeline = RagPipeline(FakeEmbedder(), llm)
    count = pipeline.index(config.DOCS_DIR, tmp_path)
    assert count >= 3
    result = pipeline.ask('Can I get a refund after purchase within seven days?')
    assert result['used_llm'] is True
    assert result['sources'][0]['source'] == 'refund_policy.md'
    assert result['grounded'] is True

def test_pipeline_skips_llm_when_nothing_matches(tmp_path):
    llm = fake_llm_factory('should never be used')
    pipeline = RagPipeline(FakeEmbedder(), llm)
    pipeline.index(config.DOCS_DIR, tmp_path)
    result = pipeline.ask('zebra xylophone quantum')
    assert result['answer'] == config.NO_ANSWER
    assert result['used_llm'] is False
    assert llm.calls == []

def test_grounding_flags_invented_answer():
    hits = [{'text': 'Refunds are available within 7 days of purchase.'}]
    good = check_grounding('Refunds are available within 7 days [1].', hits)
    bad = check_grounding('Premium members receive lifetime guarantees and free upgrades [3].', hits)
    assert good['grounded'] is True
    assert bad['grounded'] is False
    assert bad['bad_citations'] == [3]
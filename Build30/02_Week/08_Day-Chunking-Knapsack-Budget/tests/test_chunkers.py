from pathlib import Path
import pytest
from chunkers import get_chunker, STRATEGIES
from chunkers.tokens import WordCounter
from chunkers.quality import report
TEXT = Path('data/sample_policy.md').read_text(encoding='utf-8')
COUNTER = WordCounter()
@pytest.mark.parametrize('name', list(STRATEGIES))
def test_no_chunk_is_over_size(name):
    chunks = get_chunker(name, COUNTER, size=60, overlap=10).chunk(TEXT, 'policy')
    assert chunks
    assert max(c.token_count for c in chunks) <= 60
@pytest.mark.parametrize('name', list(STRATEGIES))
def test_no_text_is_lost(name):
    chunks = get_chunker(name, COUNTER, size=60, overlap=10).chunk(TEXT, 'policy')
    assert report(chunks, TEXT)['lost_chars'] == 0
@pytest.mark.parametrize('name', list(STRATEGIES))
def test_metadata_points_back_to_source(name):
    chunks = get_chunker(name, COUNTER, size=60, overlap=10).chunk(TEXT, 'policy')
    for i, c in enumerate(chunks):
        assert c.index == i
        assert TEXT[c.start_char:c.end_char] == c.text
        assert c.source == 'policy'
@pytest.mark.parametrize('name', list(STRATEGIES))
def test_overlap_zero_means_no_shared_text(name):
    chunks = get_chunker(name, COUNTER, size=60, overlap=0).chunk(TEXT, 'policy')
    for prev, cur in zip(chunks, chunks[1:]):
        assert prev.end_char <= cur.start_char
def test_overlap_shares_text_between_neighbours():
    chunks = get_chunker('sentence', COUNTER, size=60, overlap=20).chunk(TEXT, 'policy')
    assert any(prev.end_char > cur.start_char for prev, cur in zip(chunks, chunks[1:]))
def test_sentence_chunker_cuts_cleaner_than_fixed():
    fixed = report(get_chunker('fixed', COUNTER, 60, 10).chunk(TEXT, 'p'), TEXT)
    sent = report(get_chunker('sentence', COUNTER, 60, 10).chunk(TEXT, 'p'), TEXT)
    assert sent['bad_end_pct'] < fixed['bad_end_pct']
def test_headings_travel_with_chunks():
    chunks = get_chunker('recursive', COUNTER, size=60, overlap=10).chunk(TEXT, 'policy')
    refund = [c for c in chunks if 'invoice number' in c.text]
    assert refund and refund[0].heading == 'Refunds'
def test_oversize_sentence_is_split_not_dropped():
    long_sentence = 'word ' * 500
    chunks = get_chunker('sentence', COUNTER, size=50, overlap=0).chunk(long_sentence, 'long')
    assert max(c.token_count for c in chunks) <= 50
    assert sum(c.token_count for c in chunks) == 500
def test_bad_settings_are_rejected():
    with pytest.raises(ValueError):
        get_chunker('fixed', COUNTER, size=50, overlap=50)
    with pytest.raises(ValueError):
        get_chunker('nope', COUNTER)

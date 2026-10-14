from rag.chunker import chunk_text, split_sentences


def make_text():
    return ' '.join(f'Sentence number {i} has a few words.' for i in range(30))


def test_chunks_respect_word_limit():
    chunks = chunk_text(make_text(), 'sample.md', max_words=20, overlap_sentences=1)
    assert len(chunks) > 1
    assert all(len(chunk.text.split()) <= 20 for chunk in chunks)


def test_overlap_repeats_last_sentence():
    chunks = chunk_text(make_text(), 'sample.md', max_words=20, overlap_sentences=1)
    last_sentence = split_sentences(chunks[0].text)[-1]
    assert last_sentence in chunks[1].text

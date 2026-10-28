from chunker import load_chunks, split_long
def test_short_lines_are_skipped_and_source_is_file_name(tmp_path):
    note = tmp_path / 'day-09.md'
    note.write_text('# Day 09\nProblem: tickets took 3 days to answer because nobody found the policy page.\nOK\n', encoding='utf-8')
    chunks = load_chunks(tmp_path)
    assert len(chunks) == 1 and chunks[0]['source'] == 'day-09'
def test_long_text_is_split_on_sentences():
    long_text = ' '.join(['This sentence is about forty characters long.'] * 40)
    pieces = split_long(long_text)
    assert len(pieces) > 1 and all(len(p) <= 950 for p in pieces)

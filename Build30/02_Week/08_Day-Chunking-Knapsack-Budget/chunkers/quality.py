def report(chunks, text, limit=510):
    if not chunks:
        return {'chunks': 0}
    sizes = [c.token_count for c in chunks]
    clean_end = sum(1 for c in chunks if c.text[-1] in '.!?:' or c.text.splitlines()[-1].lstrip().startswith('#'))
    clean_start = sum(1 for c in chunks if not c.text[0].islower())
    covered = [False] * len(text)
    for c in chunks:
        for i in range(c.start_char, c.end_char):
            covered[i] = True
    lost = sum(1 for i, ch in enumerate(text) if not covered[i] and not ch.isspace())
    shared = 0
    total = 0
    for prev, cur in zip(chunks, chunks[1:]):
        shared += max(0, prev.end_char - cur.start_char)
    for c in chunks:
        total += c.end_char - c.start_char
    return {
        'chunks': len(chunks),
        'avg_tokens': round(sum(sizes) / len(sizes), 1),
        'max_tokens': max(sizes),
        'over_limit': sum(1 for s in sizes if s > limit),
        'bad_end_pct': round(100 * (1 - clean_end / len(chunks)), 1),
        'bad_start_pct': round(100 * (1 - clean_start / len(chunks)), 1),
        'overlap_pct': round(100 * shared / total, 1),
        'lost_chars': lost,
    }

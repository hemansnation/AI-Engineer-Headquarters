import re

SKIP = {
    'that', 'this', 'with', 'from', 'have', 'will', 'your', 'they', 'them', 'been',
    'were', 'what', 'when', 'which', 'their', 'there', 'about', 'would', 'could',
    'should', 'also', 'into', 'than', 'then', 'only', 'does', 'each',
}

def content_words(text: str) -> set[str]:
    words = re.findall(r'[a-z0-9]+', text.lower())
    return {w for w in words if len(w) > 3 and w not in SKIP}

def check_grounding(answer: str, hits: list[dict]) -> dict:
    cited = sorted({int(n) for n in re.findall(r'\[(\d+)\]', answer)})
    valid = [n for n in cited if 1 <= n <= len(hits)]
    bad = [n for n in cited if n not in valid]

    context_words: set[str] = set()
    for hit in hits:
        context_words |= content_words(hit['text'])

    answer_words = content_words(re.sub(r'\[\d+\]', '', answer))
    overlap = len(answer_words & context_words) / len(answer_words) if answer_words else 0.0

    grounded = bool(valid) and not bad and overlap >= 0.6
    return {'cited': valid, 'bad_citations': bad, 'overlap': round(overlap, 2), 'grounded': grounded}
import re
from pathlib import Path
LIMIT = 900
def split_long(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    pieces, current = [], ''
    for sentence in sentences:
        if current and len(current) + len(sentence) > LIMIT:
            pieces.append(current)
            current = ''
        current = (current + ' ' + sentence).strip()
    if current:
        pieces.append(current)
    return pieces
def load_chunks(folder):
    chunks = []
    for path in sorted(Path(folder).glob('*.md')):
        for line in path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if len(line) < 40:
                continue
            for piece in split_long(line):
                chunks.append({'source': path.stem, 'text': piece})
    return chunks

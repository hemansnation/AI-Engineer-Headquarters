import re
from dataclasses import dataclass
from pathlib import Path
from rag.config import DOCS_DIR, MAX_WORDS, OVERLAP_SENTENCES


@dataclass
class Chunk:
    chunk_id: str
    source: str
    position: int
    text: str


def split_sentences(text):
    parts = re.split(r'(?<=[.!?])\s+|\n+', text)
    sentences = []
    pending = ''
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith('#'):
            pending = (pending + ' ' + part).strip()
            continue
        sentences.append((pending + ' ' + part).strip())
        pending = ''
    if pending:
        sentences.append(pending)
    return sentences


def chunk_text(text, source, max_words=MAX_WORDS, overlap_sentences=OVERLAP_SENTENCES):
    sentences = split_sentences(text)
    chunks = []
    current = []
    count = 0
    position = 0
    for sentence in sentences:
        words = len(sentence.split())
        if current and count + words > max_words:
            chunks.append(Chunk(f'{source}#{position}', source, position, ' '.join(current)))
            position += 1
            current = current[-overlap_sentences:] if overlap_sentences else []
            count = sum(len(s.split()) for s in current)
        current.append(sentence)
        count += words
    if current:
        chunks.append(Chunk(f'{source}#{position}', source, position, ' '.join(current)))
    return chunks


def load_chunks(docs_dir=DOCS_DIR):
    chunks = []
    for path in sorted(Path(docs_dir).glob('*.md')):
        text = path.read_text(encoding='utf-8')
        chunks.extend(chunk_text(text, path.name))
    return chunks

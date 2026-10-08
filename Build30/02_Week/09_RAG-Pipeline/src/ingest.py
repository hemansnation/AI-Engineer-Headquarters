import re
from pathlib import Path

def read_file(path: Path) -> str:
    if path.suffix.lower() == '.pdf':
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return '\n'.join(page.extract_text() or '' for page in reader.pages)
    return path.read_text(encoding='utf-8')

def load_documents(folder: Path) -> list[dict]:
    docs = []
    for path in sorted(folder.rglob('*')):
        if path.suffix.lower() in {'.txt', '.md', '.pdf'}:
            text = read_file(path).strip()
            if text:
                docs.append({'source': path.name, 'text': text})
    return docs

def split_sentences(text: str) -> list[str]:
    text = re.sub(r'\s+', ' ', text)
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]

def chunk_text(text: str, source: str, max_words: int, overlap_words: int) -> list[dict]:
    sentences = split_sentences(text)
    chunks = []
    current: list[str] = []
    count = 0

    def flush():
        if current:
            chunks.append({
                'id': f'{source}#{len(chunks)}',
                'source': source,
                'position': len(chunks),
                'text': ' '.join(current),
            })

    for sentence in sentences:
        words = len(sentence.split())
        if current and count + words > max_words:
            flush()
            tail: list[str] = []
            tail_count = 0
            for old in reversed(current):
                old_words = len(old.split())
                if tail_count + old_words > overlap_words:
                    break
                tail.insert(0, old)
                tail_count += old_words
            current[:] = tail
            count = tail_count
        current.append(sentence)
        count += words
    flush()
    return chunks

def build_chunks(folder: Path, max_words: int, overlap_words: int) -> list[dict]:
    all_chunks = []
    for doc in load_documents(folder):
        all_chunks.extend(chunk_text(doc['text'], doc['source'], max_words, overlap_words))
    return all_chunks
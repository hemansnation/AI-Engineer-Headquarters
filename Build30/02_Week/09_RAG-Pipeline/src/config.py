from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / 'data' / 'docs'
INDEX_DIR = ROOT / 'index'

EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
LLM_MODEL = 'llama3.2:1b'

CHUNK_WORDS = 120
OVERLAP_WORDS = 25

TOP_K = 5
MIN_SCORE = 0.30
CONTEXT_BUDGET_WORDS = 450

NO_ANSWER = 'I could not find this in your documents.'
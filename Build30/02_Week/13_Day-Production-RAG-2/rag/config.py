import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / 'data' / 'docs'
INDEX_DIR = ROOT / 'index'

EMBED_MODEL = os.getenv('EMBED_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
LLM_MODEL = os.getenv('LLM_MODEL', 'qwen3:4b')
TOP_K = int(os.getenv('TOP_K', '3'))
CANDIDATES = int(os.getenv('CANDIDATES', '10'))
RRF_K = int(os.getenv('RRF_K', '60'))
MAX_WORDS = int(os.getenv('MAX_WORDS', '60'))
OVERLAP_SENTENCES = int(os.getenv('OVERLAP_SENTENCES', '1'))

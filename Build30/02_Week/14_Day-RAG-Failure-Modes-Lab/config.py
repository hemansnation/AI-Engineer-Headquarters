import os
from dotenv import load_dotenv
load_dotenv()
EMBED_MODEL = os.getenv('EMBED_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
RERANK_MODEL = os.getenv('RERANK_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
TOP_K = int(os.getenv('TOP_K', '5'))
NAIVE_CHARS = int(os.getenv('NAIVE_CHARS', '150'))
MAX_WORDS = int(os.getenv('MAX_WORDS', '25'))
OVERLAP = int(os.getenv('OVERLAP', '1'))
RRF_K = 60
LOG_DIR = 'logs'

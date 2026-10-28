import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
ROOT = Path(__file__).parent
OUT_DIR = ROOT / 'out'
LOG_FILE = ROOT / 'token_log.jsonl'
REPO_PATH = Path(os.getenv('REPO_PATH', '.')).expanduser()
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen3.5:4b')
EMBED_MODEL = os.getenv('EMBED_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
TARGET_ROLE = os.getenv('TARGET_ROLE', 'AI engineer who ships RAG and agent systems to production')
AUTHOR_NAME = os.getenv('AUTHOR_NAME', 'Your Name')
SESSION_GAP_MINUTES = int(os.getenv('SESSION_GAP_MINUTES', '90'))
READ_BUDGET_SECONDS = int(os.getenv('READ_BUDGET_SECONDS', '90'))
WORDS_PER_MINUTE = int(os.getenv('WORDS_PER_MINUTE', '230'))
MAX_RETRIES = int(os.getenv('MAX_RETRIES', '2'))
DAY_PATTERN = r'(?i)day[-_ ]?0*(\d{1,2})'

import os
from dotenv import load_dotenv
load_dotenv()
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
MAIN_MODEL = os.getenv('MAIN_MODEL', 'qwen3:4b')
GUARD_MODEL = os.getenv('GUARD_MODEL', 'qwen3:1.7b')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
SEMANTIC_THRESHOLD = float(os.getenv('SEMANTIC_THRESHOLD', '0.60'))
MAX_INPUT_CHARS = int(os.getenv('MAX_INPUT_CHARS', '4000'))
LOG_FILE = os.getenv('LOG_FILE', 'guard_log.jsonl')

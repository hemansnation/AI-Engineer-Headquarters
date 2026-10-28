import os
from dotenv import load_dotenv
load_dotenv()
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen3.5:4b')
EMBED_MODEL = os.getenv('EMBED_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
DB_PATH = os.getenv('DB_PATH', 'data/lakeview.db')
MAX_ROWS = int(os.getenv('MAX_ROWS', '200'))
ROWS_FOR_MODEL = int(os.getenv('ROWS_FOR_MODEL', '20'))
QUERY_TIMEOUT_SECONDS = float(os.getenv('QUERY_TIMEOUT_SECONDS', '2'))
TOP_K_TABLES = int(os.getenv('TOP_K_TABLES', '3'))
MAX_TRIES = int(os.getenv('MAX_TRIES', '3'))

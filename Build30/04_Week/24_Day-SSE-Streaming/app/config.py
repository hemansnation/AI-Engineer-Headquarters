import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
PROVIDER = os.getenv('PROVIDER', 'fake')
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2:3b')
FAKE_FIRST_DELAY = float(os.getenv('FAKE_FIRST_DELAY', '0.4'))
FAKE_GAP = float(os.getenv('FAKE_GAP', '0.04'))
LOG_FILE = ROOT / os.getenv('LOG_FILE', 'logs/stream_log.jsonl')

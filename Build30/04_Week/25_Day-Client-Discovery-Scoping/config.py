import os
from dotenv import load_dotenv
load_dotenv()
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen3.5:4b')
EMBED_MODEL = os.getenv('EMBED_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
HOURLY_RATE_USD = float(os.getenv('HOURLY_RATE_USD', '40'))
OVERHEAD_PCT = float(os.getenv('OVERHEAD_PCT', '0.15'))
RISK_HOLDBACK = float(os.getenv('RISK_HOLDBACK', '0.15'))
HOSTING_USD_PER_MONTH = float(os.getenv('HOSTING_USD_PER_MONTH', '150'))
TOKENS_PER_TASK = int(os.getenv('TOKENS_PER_TASK', '1500'))
EVIDENCE_THRESHOLD = float(os.getenv('EVIDENCE_THRESHOLD', '0.35'))
MAX_PAYBACK_MONTHS = float(os.getenv('MAX_PAYBACK_MONTHS', '12'))

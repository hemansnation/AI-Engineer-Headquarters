import os
from dotenv import load_dotenv
load_dotenv()
MODEL = os.getenv('OLLAMA_MODEL', 'qwen3:4b')
REVIEWER_MODEL = os.getenv('REVIEWER_MODEL', MODEL)
HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
MAX_ROUNDS = int(os.getenv('MAX_ROUNDS', '3'))
TOKEN_BUDGET = int(os.getenv('TOKEN_BUDGET', '6000'))
STALL_THRESHOLD = float(os.getenv('STALL_THRESHOLD', '0.97'))
EMBED_MODEL = os.getenv('EMBED_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')

import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
RUNS = DATA / 'runs'
GOLDEN = DATA / 'golden_v1.jsonl'
TARGET_MODEL = os.getenv('TARGET_MODEL', 'qwen3:4b')
JUDGE_MODEL = os.getenv('JUDGE_MODEL', 'gemma4')
EMBED_MODEL = os.getenv('EMBED_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
SEED = int(os.getenv('SEED', '7'))
EDIT_PASS = 0.85
EMBED_PASS = 0.75
MAX_DROP = 0.02

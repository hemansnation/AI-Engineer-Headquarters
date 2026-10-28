import json
import time
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from config import LOG_FILE, OLLAMA_MODEL
def append_record(path: Path, record: dict) -> None:
    with open(path, 'a', encoding='utf-8') as file:
        file.write(json.dumps(record) + '\n')
def track_tokens(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        response = func(*args, **kwargs)
        seconds = round(time.perf_counter() - start, 2)
        record = {
            'time': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'model': OLLAMA_MODEL,
            'input_tokens': getattr(response, 'prompt_eval_count', 0) or 0,
            'output_tokens': getattr(response, 'eval_count', 0) or 0,
            'latency_seconds': seconds,
            'paid_equivalent_usd': 0.0,
        }
        append_record(LOG_FILE, record)
        return response
    return wrapper
def running_totals(path: Path = LOG_FILE) -> dict:
    totals = {'calls': 0, 'tokens': 0, 'seconds': 0.0}
    if not path.exists():
        return totals
    with open(path, encoding='utf-8') as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            totals['calls'] += 1
            totals['tokens'] += record['input_tokens'] + record['output_tokens']
            totals['seconds'] += record['latency_seconds']
    totals['seconds'] = round(totals['seconds'], 2)
    return totals

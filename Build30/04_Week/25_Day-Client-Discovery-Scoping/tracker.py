import json
import time
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
LOG_FILE = Path('output/token_log.jsonl')
def append_record(record):
    LOG_FILE.parent.mkdir(exist_ok=True)
    with LOG_FILE.open('a', encoding='utf-8') as file:
        file.write(json.dumps(record) + '\n')
def track_tokens(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        reply = func(*args, **kwargs)
        append_record({'time': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'model': reply.model, 'input_tokens': reply.prompt_eval_count or 0, 'output_tokens': reply.eval_count or 0, 'latency_seconds': round(time.perf_counter() - start, 2)})
        return reply
    return wrapper
def running_totals():
    totals = {'calls': 0, 'tokens': 0}
    if not LOG_FILE.exists():
        return totals
    with LOG_FILE.open(encoding='utf-8') as file:
        for line in file:
            if line.strip():
                record = json.loads(line)
                totals['calls'] += 1
                totals['tokens'] += record['input_tokens'] + record['output_tokens']
    return totals

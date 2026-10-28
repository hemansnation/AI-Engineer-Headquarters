import json
import time
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
LOG_FILE = Path('token_log.jsonl')
def track(function):
    @wraps(function)
    def wrapper(prompt, *args, **kwargs):
        start = time.perf_counter()
        reply = function(prompt, *args, **kwargs)
        record = {
            'time': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'prompt_preview': prompt[:40],
            'input_tokens': getattr(reply, 'prompt_eval_count', 0) or 0,
            'output_tokens': getattr(reply, 'eval_count', 0) or 0,
            'seconds': round(time.perf_counter() - start, 2),
            'paid_equivalent_usd': 0.0,
        }
        with LOG_FILE.open('a', encoding='utf-8') as file:
            file.write(json.dumps(record) + '\n')
        return reply
    return wrapper

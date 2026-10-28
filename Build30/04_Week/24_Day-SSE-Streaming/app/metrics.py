import json
from datetime import datetime, timezone
from app import config
def log_stream(prompt, ttft, total, tokens, status):
    record = {
        'time': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'prompt_preview': prompt[:40],
        'ttft_ms': None if ttft is None else round(ttft * 1000),
        'total_ms': round(total * 1000),
        'tokens': tokens,
        'status': status,
    }
    path = config.LOG_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as file:
        file.write(json.dumps(record) + '\n')
    return record

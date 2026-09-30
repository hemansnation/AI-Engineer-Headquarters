import json
import time
from pathlib import Path

LOG_FILE = Path('logs') / 'token_log.jsonl'


def log_event(kind, text, counts):
    LOG_FILE.parent.mkdir(exist_ok=True)
    entry = {
        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'kind': kind,
        'chars': len(text),
        'utf8_bytes': len(text.encode('utf-8')),
        'counts': counts,
    }
    with LOG_FILE.open('a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    return entry
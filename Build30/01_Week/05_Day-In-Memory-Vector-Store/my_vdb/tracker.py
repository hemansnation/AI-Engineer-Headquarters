import time
from functools import wraps
from pathlib import Path

LOG_FILE = Path('logs') / 'day05.log'

def track(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        ms = (time.perf_counter() - start) * 1000
        LOG_FILE.parent.mkdir(exist_ok=True)
        with LOG_FILE.open('a', encoding='utf-8') as f:
            f.write(f'{fn.__qualname__} took {ms:.2f} ms\n')
        return result
    return wrapper
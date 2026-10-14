import time
from functools import wraps
TIMINGS = {}
def timed(stage):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            TIMINGS[stage] = round((time.perf_counter() - start) * 1000, 2)
            return result
        return wrapper
    return decorator
def reset():
    TIMINGS.clear()
def report():
    for stage, ms in TIMINGS.items():
        print(f'  {stage:<10} {ms:>8} ms')

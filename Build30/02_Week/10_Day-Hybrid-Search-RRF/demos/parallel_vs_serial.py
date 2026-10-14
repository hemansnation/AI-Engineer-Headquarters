import time
from concurrent.futures import ThreadPoolExecutor
def slow_vector_search():
    time.sleep(0.5)
    return ['d19', 'd16']
def slow_keyword_search():
    time.sleep(0.3)
    return ['d19', 'd15']
start = time.perf_counter()
slow_vector_search()
slow_keyword_search()
print(f'serial   : {time.perf_counter() - start:.2f} s')
start = time.perf_counter()
with ThreadPoolExecutor(max_workers=2) as pool:
    a = pool.submit(slow_vector_search)
    b = pool.submit(slow_keyword_search)
    a.result()
    b.result()
print(f'parallel : {time.perf_counter() - start:.2f} s')

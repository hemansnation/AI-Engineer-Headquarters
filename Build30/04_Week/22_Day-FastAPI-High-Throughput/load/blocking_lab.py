import asyncio
import sys
import time
import httpx
async def main(path: str, count: int):
    async with httpx.AsyncClient(base_url='http://127.0.0.1:8000', timeout=60) as client:
        started = time.perf_counter()
        responses = await asyncio.gather(*[client.get(path) for _ in range(count)])
        total = time.perf_counter() - started
    codes = sorted({response.status_code for response in responses})
    print(f'{path}: {count} requests finished in {total:.2f}s, status codes {codes}')
if __name__ == '__main__':
    path = sys.argv[1]
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    asyncio.run(main(path, count))

import asyncio
import time
from fastapi import APIRouter
router = APIRouter(prefix='/lab', tags=['lab'])
@router.get('/blocking')
async def blocking():
    time.sleep(1)
    return {'style': 'async def with time.sleep', 'slept': 1}
@router.get('/non-blocking')
async def non_blocking():
    await asyncio.sleep(1)
    return {'style': 'async def with await asyncio.sleep', 'slept': 1}
@router.get('/threadpool')
def threadpool():
    time.sleep(1)
    return {'style': 'plain def with time.sleep', 'slept': 1}

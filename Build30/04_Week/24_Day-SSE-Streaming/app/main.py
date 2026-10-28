import asyncio
import time
import anyio
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field
from app import config
from app.metrics import log_stream
from app.providers import get_provider
class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
app = FastAPI(title='Day 24 streaming service')
app.state.provider = get_provider()
@app.get('/')
async def home():
    return FileResponse(config.ROOT / 'static' / 'index.html')
@app.get('/health')
async def health():
    return {'status': 'ok', 'provider': app.state.provider.name}
@app.get('/stats')
async def stats():
    return app.state.provider.stats()
@app.post('/stream', response_class=EventSourceResponse)
async def stream(body: ChatRequest):
    started = time.perf_counter()
    ttft = None
    count = 0
    status = 'finished'
    source = app.state.provider.stream(body.prompt)
    try:
        async for token in source:
            if ttft is None:
                ttft = time.perf_counter() - started
            count += 1
            yield ServerSentEvent(raw_data=token, event='token', id=str(count))
        yield ServerSentEvent(raw_data=str(count), event='done')
    except asyncio.CancelledError:
        status = 'client_left'
        raise
    except Exception as error:
        status = 'error'
        yield ServerSentEvent(raw_data=f'upstream failed: {type(error).__name__}', event='error')
    finally:
        with anyio.CancelScope(shield=True):
            await source.aclose()
        log_stream(body.prompt, ttft, time.perf_counter() - started, count, status)

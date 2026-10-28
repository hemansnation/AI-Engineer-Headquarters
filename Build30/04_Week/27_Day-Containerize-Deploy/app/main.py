import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

from app.config import settings
from app.embedder import Embedder
from app.llm import ask_llm, llm_enabled
from app.logging_setup import setup_logging
from app.store import VectorStore

setup_logging(settings.log_level)
log = logging.getLogger('service')


def build_store():
    embedder = Embedder()
    store = VectorStore(embedder)
    store.load(settings.docs_path)
    return store


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.store = None

    async def load():
        started = time.perf_counter()
        try:
            app.state.store = await asyncio.to_thread(build_store)
        except Exception:
            log.exception('startup failed while loading the model or documents')
            return
        log.info('ready', extra={'fields': {
            'docs': len(app.state.store.texts),
            'seconds': round(time.perf_counter() - started, 2),
            'env': settings.app_env,
            'llm_enabled': llm_enabled(),
        }})

    task = asyncio.create_task(load())
    yield
    task.cancel()
    log.info('shutdown')


app = FastAPI(title='Build30 Day 27', lifespan=lifespan)


@app.middleware('http')
async def log_requests(request: Request, call_next):
    request_id = uuid.uuid4().hex[:8]
    started = time.perf_counter()
    response = await call_next(request)
    response.headers['x-request-id'] = request_id
    log.info('request', extra={'fields': {
        'request_id': request_id,
        'method': request.method,
        'path': request.url.path,
        'status': response.status_code,
        'ms': round((time.perf_counter() - started) * 1000, 1),
    }})
    return response


def get_store():
    store = app.state.store
    if store is None:
        raise HTTPException(status_code=503, detail='model is still loading, try again in a few seconds')
    return store


@app.get('/')
def root():
    return {'service': 'build30-day27', 'docs': '/docs', 'health': '/health', 'ready': '/ready'}


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.get('/ready')
def ready(response: Response):
    store = app.state.store
    if store is None:
        response.status_code = 503
        return {'ready': False}
    return {'ready': True, 'docs': len(store.texts)}


@app.get('/stats')
def stats():
    store = get_store()
    return {
        'docs': len(store.texts),
        'embedding_cache': store.embedder.cache.stats(),
        'llm_enabled': llm_enabled(),
        'env': settings.app_env,
    }


@app.get('/search')
def search(q: str = Query(min_length=2, max_length=300), k: int = Query(default=settings.top_k, ge=1, le=10)):
    store = get_store()
    return {'query': q, 'results': store.search(q, k)}


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)


@app.post('/ask')
async def ask(body: AskRequest):
    store = get_store()
    hits = await asyncio.to_thread(store.search, body.question, settings.top_k)
    sources = [hit['text'] for hit in hits]
    fallback = {'mode': 'retrieval-only', 'answer': sources[0] if sources else '', 'sources': hits}
    if not llm_enabled():
        return fallback
    try:
        answer = await ask_llm(body.question, sources)
    except (httpx.HTTPError, KeyError, IndexError) as error:
        log.warning('llm failed, using retrieval only', extra={'fields': {'error': type(error).__name__}})
        return fallback
    return {'mode': 'llm', 'answer': answer, 'sources': hits}

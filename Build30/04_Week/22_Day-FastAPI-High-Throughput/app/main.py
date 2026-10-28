import asyncio
import math
import time
import uuid
from contextlib import asynccontextmanager
import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from app import lab
from app.config import get_settings
from app.errors import (
    ServiceError,
    service_error_handler,
    unhandled_error_handler,
    validation_error_handler,
)
from app.limiter import TokenBucketLimiter
from app.llm import MockLLM, OllamaLLM, guarded_generate
from app.pipeline import Retriever
from app.schemas import AskRequest, AskResponse, SourceChunk, Usage
from app.usage import append_usage
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.retriever = await asyncio.to_thread(Retriever, settings.embed_model, settings.corpus_path)
    app.state.http = httpx.AsyncClient(timeout=settings.llm_timeout_seconds)
    if settings.llm_mode == 'ollama':
        app.state.llm = OllamaLLM(settings.ollama_url, settings.ollama_model, app.state.http)
    else:
        app.state.llm = MockLLM()
    app.state.gate = asyncio.Semaphore(settings.max_concurrent_llm_calls)
    app.state.limiter = TokenBucketLimiter(settings.rate_limit_capacity, settings.rate_limit_refill_per_second)
    app.state.jobs = {}
    yield
    await app.state.http.aclose()
app = FastAPI(title='Build30 Day 22 RAG API', version='1.0.0', lifespan=lifespan)
app.add_exception_handler(ServiceError, service_error_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)
app.add_exception_handler(Exception, unhandled_error_handler)
app.include_router(lab.router)
@app.middleware('http')
async def add_request_id(request: Request, call_next):
    request.state.request_id = uuid.uuid4().hex[:12]
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers['X-Request-ID'] = request.state.request_id
    response.headers['X-Process-Time-Ms'] = f'{elapsed_ms:.1f}'
    return response
async def rate_limit(request: Request):
    key = request.headers.get('x-api-key') or (request.client.host if request.client else 'anonymous')
    allowed, wait_seconds = request.app.state.limiter.allow(key)
    if not allowed:
        retry_after = max(1, math.ceil(wait_seconds))
        raise ServiceError(
            429,
            'rate_limited',
            f'Too many requests. Try again in {retry_after} seconds.',
            {'Retry-After': str(retry_after)},
        )
@app.get('/health')
async def health():
    return {'status': 'ok'}
@app.get('/ready')
async def ready(request: Request):
    state = request.app.state
    settings = get_settings()
    if getattr(state, 'retriever', None) is None:
        raise ServiceError(503, 'not_ready', 'Embedding model is not loaded yet.')
    if settings.llm_mode == 'ollama':
        try:
            response = await state.http.get(f'{settings.ollama_url}/api/tags', timeout=2.0)
            response.raise_for_status()
        except httpx.HTTPError:
            raise ServiceError(503, 'not_ready', 'Ollama is not reachable.')
    return {'status': 'ready', 'llm_mode': settings.llm_mode, 'chunks': state.retriever.size}
@app.post('/ask', response_model=AskResponse, dependencies=[Depends(rate_limit)])
async def ask(body: AskRequest, request: Request, background: BackgroundTasks):
    state = request.app.state
    settings = get_settings()
    started = time.perf_counter()
    top_k = body.top_k or settings.top_k
    hits = await state.retriever.search(body.question, top_k)
    context = '\n'.join(f'- {text}' for _, text, _ in hits)
    prompt = (
        'Answer using only the context below. If the answer is not in the context, say you do not know.\n\n'
        f'Context:\n{context}\n\nQuestion: {body.question}\nAnswer:'
    )
    result = await guarded_generate(state.llm, state.gate, prompt, settings.queue_wait_seconds)
    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    request_id = request.state.request_id
    background.add_task(
        append_usage,
        settings.usage_log_path,
        {
            'request_id': request_id,
            'prompt_tokens': result.prompt_tokens,
            'completion_tokens': result.completion_tokens,
            'latency_ms': latency_ms,
        },
    )
    return AskResponse(
        request_id=request_id,
        answer=result.text,
        sources=[SourceChunk(id=i, text=text, score=round(score, 4)) for i, text, score in hits],
        usage=Usage(prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens),
        latency_ms=latency_ms,
    )
async def run_reindex(app: FastAPI, job_id: str):
    try:
        await asyncio.to_thread(app.state.retriever.reload)
        app.state.jobs[job_id] = {'status': 'done', 'chunks': app.state.retriever.size}
    except Exception as error:
        app.state.jobs[job_id] = {'status': 'failed', 'detail': str(error)}
@app.post('/reindex', status_code=202)
async def reindex(request: Request, background: BackgroundTasks):
    job_id = uuid.uuid4().hex[:8]
    request.app.state.jobs[job_id] = {'status': 'running'}
    background.add_task(run_reindex, request.app, job_id)
    return {'job_id': job_id, 'status': 'running'}
@app.get('/jobs/{job_id}')
async def job_status(job_id: str, request: Request):
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise ServiceError(404, 'not_found', 'No job with that id.')
    return job

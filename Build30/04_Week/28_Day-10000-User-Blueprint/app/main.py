import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from cache.semantic_cache import SemanticCache, HashEmbedder
from cqrs.service import Store, WriteSide, ReadSide
from gateway.token_bucket import PerUserLimiter
from routing.hash_ring import HashRing
from app.llm import fake_llm, ollama_llm
app = FastAPI(title='Day 28 blueprint')
store = Store()
cache = SemanticCache(HashEmbedder(), threshold=0.9)
limiter = PerUserLimiter(capacity=5, refill_per_sec=1)
ring = HashRing(['worker-1', 'worker-2', 'worker-3', 'worker-4'])
llm = ollama_llm if os.getenv('USE_OLLAMA') == '1' else fake_llm
writer = WriteSide(store, cache)
reader = ReadSide(store, cache, llm)
class UploadBody(BaseModel):
    tenant: str
    doc_id: str
    text: str
class AskBody(BaseModel):
    tenant: str
    user_id: str
    question: str
@app.get('/health')
def health():
    return {'status': 'ok', 'cache_hit_rate': round(cache.hit_rate(), 3)}
@app.post('/documents')
def upload(body: UploadBody):
    version = writer.upload(body.tenant, body.doc_id, body.text)
    return {'tenant': body.tenant, 'version': version}
@app.post('/ask')
def ask(body: AskBody):
    if not limiter.allow(body.user_id):
        raise HTTPException(status_code=429, detail='slow down')
    result = reader.ask(body.tenant, body.question)
    result['worker'] = ring.get_node(body.tenant)
    return result

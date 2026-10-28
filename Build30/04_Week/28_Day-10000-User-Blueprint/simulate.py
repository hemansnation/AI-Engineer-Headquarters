import random
from collections import Counter
from cache.semantic_cache import SemanticCache, HashEmbedder
from cqrs.service import Store, WriteSide, ReadSide
from gateway.token_bucket import PerUserLimiter
from routing.hash_ring import HashRing
from app.llm import fake_llm
random.seed(28)
store = Store()
cache = SemanticCache(HashEmbedder(), threshold=0.9)
limiter = PerUserLimiter(capacity=5, refill_per_sec=1)
ring = HashRing(['worker-1', 'worker-2', 'worker-3', 'worker-4'])
writer = WriteSide(store, cache)
reader = ReadSide(store, cache, fake_llm)
tenants = [f'company-{i}' for i in range(50)]
questions = [f'how do I apply for leave type {i}' for i in range(40)]
for t in tenants:
    writer.upload(t, 'policy', 'leave policy text')
sources = Counter()
workers = Counter()
blocked = 0
for user in range(10000):
    tenant = tenants[user % len(tenants)]
    burst = 20 if user == 7 else 3
    for _ in range(burst):
        if not limiter.allow(f'user-{user}'):
            blocked += 1
            continue
        result = reader.ask(tenant, random.choice(questions))
        sources[result['source']] += 1
        workers[ring.get_node(tenant)] += 1
print('served', dict(sources))
print('blocked by limiter', blocked)
print('cache hit rate', round(cache.hit_rate(), 3))
print('requests per worker', dict(workers))

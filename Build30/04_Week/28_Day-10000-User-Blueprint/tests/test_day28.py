import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from capacity.estimate import estimate_cluster_capacity
from gateway.token_bucket import TokenBucket, PerUserLimiter
from cache.semantic_cache import SemanticCache, HashEmbedder
from routing.hash_ring import HashRing
from cqrs.service import Store, WriteSide, ReadSide
from app.main import app
def test_capacity():
    r = estimate_cluster_capacity(10000, 20, 1500)
    assert r.daily_queries == 60000
    assert r.daily_tokens == 90000000
    assert r.average_qps == 2.08
    assert r.peak_qps == 10.42
    assert r.gpus_needed == 11
    cached = estimate_cluster_capacity(10000, 20, 1500, cache_hit_rate=0.3)
    assert cached.gpus_needed == 8
    assert cached.monthly_cost_usd < r.monthly_cost_usd
def test_bucket():
    t = [0.0]
    b = TokenBucket(3, 1, clock=lambda: t[0])
    assert [b.allow() for _ in range(4)] == [True, True, True, False]
    t[0] = 2.0
    assert b.allow() and b.allow() and not b.allow()
def test_per_user():
    t = [0.0]
    limiter = PerUserLimiter(1, 0.1, clock=lambda: t[0])
    assert limiter.allow('a') and not limiter.allow('a') and limiter.allow('b')
def test_ring_moves_only_a_share():
    ring = HashRing(['n1', 'n2', 'n3'])
    keys = [f'tenant-{i}' for i in range(2000)]
    before = {k: ring.get_node(k) for k in keys}
    ring.add_node('n4')
    moved = sum(1 for k in keys if ring.get_node(k) != before[k])
    assert moved / len(keys) < 0.40
def test_cache_and_cqrs():
    cache = SemanticCache(HashEmbedder(), threshold=0.9)
    store = Store()
    calls = []
    def llm(q, docs):
        calls.append(q)
        return 'answer for ' + q
    w, r = WriteSide(store, cache), ReadSide(store, cache, llm)
    w.upload('acme', 'd1', 'leave policy')
    assert r.ask('acme', 'how many leave days')['source'] == 'model'
    assert r.ask('acme', 'how many leave days')['source'] == 'cache'
    assert r.ask('other', 'how many leave days')['source'] == 'model'
    w.upload('acme', 'd2', 'new policy')
    assert r.ask('acme', 'how many leave days')['source'] == 'model'
    assert len(calls) == 3
def test_api():
    client = TestClient(app)
    assert client.get('/health').json()['status'] == 'ok'
    assert client.post('/documents', json={'tenant': 'acme', 'doc_id': 'd1', 'text': 'x'}).json()['version'] == 1
    body = {'tenant': 'acme', 'user_id': 'u1', 'question': 'what is the leave policy'}
    assert client.post('/ask', json=body).json()['source'] == 'model'
    assert client.post('/ask', json=body).json()['source'] == 'cache'
    codes = [client.post('/ask', json=body).status_code for _ in range(8)]
    assert 429 in codes

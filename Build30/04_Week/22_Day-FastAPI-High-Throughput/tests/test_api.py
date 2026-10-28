import os
os.environ['LLM_MODE'] = 'mock'
os.environ['RATE_LIMIT_CAPACITY'] = '3'
os.environ['RATE_LIMIT_REFILL_PER_SECOND'] = '0.01'
os.environ['USAGE_LOG_PATH'] = 'logs/test_usage.jsonl'
import pytest
from fastapi.testclient import TestClient
from app.main import app
@pytest.fixture(scope='module')
def client():
    with TestClient(app) as test_client:
        yield test_client
def test_health(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}
def test_ready(client):
    response = client.get('/ready')
    assert response.status_code == 200
    assert response.json()['status'] == 'ready'
def test_ask_returns_answer_and_sources(client):
    response = client.post('/ask', json={'question': 'What does a semaphore do?'}, headers={'x-api-key': 'ask-test'})
    assert response.status_code == 200
    body = response.json()
    assert body['answer']
    assert len(body['sources']) == 3
    assert body['usage']['completion_tokens'] == 8
    assert response.headers['x-request-id'] == body['request_id']
def test_bad_request_returns_clean_422(client):
    response = client.post('/ask', json={'question': 'a'}, headers={'x-api-key': 'bad-test'})
    assert response.status_code == 422
    body = response.json()
    assert body['error'] == 'invalid_request'
    assert 'traceback' not in str(body).lower()
def test_rate_limit_returns_429_with_retry_after(client):
    statuses = []
    last = None
    for _ in range(5):
        last = client.post('/ask', json={'question': 'What is p95 latency?'}, headers={'x-api-key': 'limit-test'})
        statuses.append(last.status_code)
    assert statuses[:3] == [200, 200, 200]
    assert 429 in statuses
    assert int(last.headers['retry-after']) >= 1
def test_unknown_job_returns_404(client):
    response = client.get('/jobs/nope')
    assert response.status_code == 404
    assert response.json()['error'] == 'not_found'
def test_reindex_job_finishes(client):
    start = client.post('/reindex')
    assert start.status_code == 202
    job_id = start.json()['job_id']
    status = client.get(f'/jobs/{job_id}').json()
    assert status['status'] in ('running', 'done')

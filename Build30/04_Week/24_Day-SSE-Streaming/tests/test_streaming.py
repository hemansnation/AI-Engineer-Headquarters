import asyncio
import json
import socket
import threading
import time
import httpx
import pytest
import uvicorn
from app import config
from app.main import app
from app.providers import FakeProvider, OllamaProvider
def parse_sse(text):
    events = []
    for frame in text.split('\n\n'):
        event = None
        data = []
        for line in frame.split('\n'):
            if line.startswith(':') or ':' not in line:
                continue
            field, value = line.split(':', 1)
            value = value[1:] if value.startswith(' ') else value
            if field == 'event':
                event = value
            if field == 'data':
                data.append(value)
        if event:
            events.append((event, '\n'.join(data)))
    return events
@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'LOG_FILE', tmp_path / 'stream_log.jsonl')
    app.state.provider = FakeProvider(first_delay=0, gap=0)
    return tmp_path / 'stream_log.jsonl'
async def test_stream_returns_tokens_then_done(isolated):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
        response = await client.post('/stream', json={'prompt': 'hello'})
    assert response.headers['content-type'].startswith('text/event-stream')
    events = parse_sse(response.text)
    tokens = [data for name, data in events if name == 'token']
    assert ''.join(tokens).startswith(' You asked: hello.')
    assert events[-1][0] == 'done'
    assert events[-1][1] == str(len(tokens))
    record = json.loads(isolated.read_text().splitlines()[-1])
    assert record['status'] == 'finished'
    assert record['tokens'] == len(tokens)
    assert record['ttft_ms'] is not None
async def test_empty_prompt_is_rejected():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
        response = await client.post('/stream', json={'prompt': ''})
    assert response.status_code == 422
async def test_closing_the_generator_closes_the_upstream():
    provider = FakeProvider(first_delay=0, gap=0)
    source = provider.stream('hello')
    await source.__anext__()
    assert provider.stats() == {'active': 1, 'closed': 0}
    await source.aclose()
    assert provider.stats() == {'active': 0, 'closed': 1}
async def test_ollama_provider_reads_ndjson_lines():
    lines = [
        {'message': {'role': 'assistant', 'content': 'Hel'}, 'done': False},
        {'message': {'role': 'assistant', 'content': 'lo'}, 'done': False},
        {'message': {'role': 'assistant', 'content': ''}, 'done': True},
    ]
    body = ''.join(json.dumps(line) + '\n' for line in lines)
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=body.encode()))
    provider = OllamaProvider('http://ollama.test', 'any-model', transport=transport)
    tokens = [token async for token in provider.stream('hi')]
    assert tokens == ['Hel', 'lo']
    assert provider.stats() == {'active': 0, 'closed': 1}
async def test_upstream_failure_becomes_an_error_event():
    def refuse(request):
        raise httpx.ConnectError('model server is down')
    provider = OllamaProvider('http://ollama.test', 'any-model', transport=httpx.MockTransport(refuse))
    app.state.provider = provider
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
        response = await client.post('/stream', json={'prompt': 'hello'})
    events = parse_sse(response.text)
    assert events[-1] == ('error', 'upstream failed: ConnectError')
@pytest.fixture
def live_server():
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, log_level='warning'))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.02)
    yield f'http://127.0.0.1:{port}'
    server.should_exit = True
    thread.join(timeout=5)
async def test_client_leaving_closes_the_upstream(live_server):
    app.state.provider = FakeProvider(first_delay=0, gap=0.02)
    async with httpx.AsyncClient(base_url=live_server, timeout=5) as client:
        seen = 0
        async with client.stream('POST', '/stream', json={'prompt': 'hello'}) as response:
            async for line in response.aiter_lines():
                if line.startswith('event: token'):
                    seen += 1
                if seen == 3:
                    break
        stats = {}
        for _ in range(50):
            stats = (await client.get('/stats')).json()
            if stats['active'] == 0:
                break
            await asyncio.sleep(0.05)
    assert stats == {'active': 0, 'closed': 1}
    last = json.loads(config.LOG_FILE.read_text().splitlines()[-1])
    assert last['status'] == 'client_left'

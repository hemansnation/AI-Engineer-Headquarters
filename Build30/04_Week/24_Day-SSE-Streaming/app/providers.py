import asyncio
import json
import httpx
from app import config
REPLY = (
    'Streaming means you read the first word while the model is still writing the rest. '
    'The server sends each token the moment it exists, and the browser paints it right away. '
    'Nothing waits for the full answer, so the wait you feel is the time to the first token.'
)
class FakeProvider:
    name = 'fake'
    def __init__(self, first_delay=0.4, gap=0.04):
        self.first_delay = first_delay
        self.gap = gap
        self.active = 0
        self.closed = 0
    async def stream(self, prompt):
        self.active += 1
        try:
            await asyncio.sleep(self.first_delay)
            for word in (f'You asked: {prompt}. ' + REPLY).split(' '):
                yield ' ' + word
                await asyncio.sleep(self.gap)
        finally:
            self.active -= 1
            self.closed += 1
    def stats(self):
        return {'active': self.active, 'closed': self.closed}
class OllamaProvider:
    name = 'ollama'
    def __init__(self, base_url, model, transport=None):
        self.base_url = base_url
        self.model = model
        self.transport = transport
        self.active = 0
        self.closed = 0
    async def stream(self, prompt):
        payload = {
            'model': self.model,
            'messages': [{'role': 'user', 'content': prompt}],
            'stream': True,
        }
        self.active += 1
        try:
            timeout = httpx.Timeout(120.0, connect=5.0)
            async with httpx.AsyncClient(base_url=self.base_url, timeout=timeout, transport=self.transport) as client:
                async with client.stream('POST', '/api/chat', json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line)
                        text = chunk.get('message', {}).get('content', '')
                        if text:
                            yield text
                        if chunk.get('done'):
                            break
        finally:
            self.active -= 1
            self.closed += 1
    def stats(self):
        return {'active': self.active, 'closed': self.closed}
def get_provider():
    if config.PROVIDER == 'ollama':
        return OllamaProvider(config.OLLAMA_URL, config.OLLAMA_MODEL)
    return FakeProvider(config.FAKE_FIRST_DELAY, config.FAKE_GAP)

import asyncio
from dataclasses import dataclass
import httpx
from app.errors import ServiceError
@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
class MockLLM:
    async def generate(self, prompt: str) -> LLMResult:
        await asyncio.sleep(0.5)
        return LLMResult(
            text='Mock answer built from the context you sent.',
            prompt_tokens=len(prompt.split()),
            completion_tokens=8,
        )
class OllamaLLM:
    def __init__(self, base_url: str, model: str, client: httpx.AsyncClient):
        self.base_url = base_url
        self.model = model
        self.client = client
    async def generate(self, prompt: str) -> LLMResult:
        payload = {
            'model': self.model,
            'messages': [{'role': 'user', 'content': prompt}],
            'stream': False,
        }
        response = await self.client.post(f'{self.base_url}/api/chat', json=payload)
        response.raise_for_status()
        data = response.json()
        return LLMResult(
            text=data['message']['content'],
            prompt_tokens=data.get('prompt_eval_count', 0),
            completion_tokens=data.get('eval_count', 0),
        )
async def guarded_generate(llm, gate: asyncio.Semaphore, prompt: str, wait_seconds: float) -> LLMResult:
    try:
        await asyncio.wait_for(gate.acquire(), timeout=wait_seconds)
    except TimeoutError:
        raise ServiceError(503, 'busy', 'Too many requests in progress. Try again in a few seconds.', {'Retry-After': '2'})
    try:
        return await llm.generate(prompt)
    except httpx.TimeoutException:
        raise ServiceError(504, 'llm_timeout', 'The model took too long to answer.')
    except httpx.ConnectError:
        raise ServiceError(503, 'llm_unavailable', 'Cannot reach the local model server. Is Ollama running?')
    except httpx.HTTPStatusError as error:
        raise ServiceError(502, 'llm_error', f'Model server returned status {error.response.status_code}.')
    finally:
        gate.release()

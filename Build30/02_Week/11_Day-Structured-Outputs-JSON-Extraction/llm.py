import json
import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

TICKS = chr(96) * 3

@dataclass
class LLMReply:
    text: str
    input_tokens: int
    output_tokens: int
    seconds: float

class GeminiProvider:
    def __init__(self):
        from google import genai

        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError('GEMINI_API_KEY is missing in your .env file')
        self.model = os.getenv('GEMINI_MODEL', 'gemini-3.1-flash-lite')
        self.client = genai.Client(api_key=api_key)

    def ask(self, prompt, schema=None):
        start = time.perf_counter()
        options = {}
        if schema is not None:
            options['response_format'] = {
                'type': 'text',
                'mime_type': 'application/json',
                'schema': schema,
            }
        interaction = self.client.interactions.create(
            model=self.model, input=prompt, **options
        )
        usage = getattr(interaction, 'usage', None)
        return LLMReply(
            text=interaction.output_text or '',
            input_tokens=int(getattr(usage, 'total_input_tokens', 0) or 0),
            output_tokens=int(getattr(usage, 'total_output_tokens', 0) or 0),
            seconds=round(time.perf_counter() - start, 2),
        )

class OllamaProvider:
    def __init__(self):
        from ollama import Client

        self.model = os.getenv('OLLAMA_MODEL', 'gemma4')
        self.client = Client(host=os.getenv('OLLAMA_HOST', 'http://localhost:11434'))

    def ask(self, prompt, schema=None):
        start = time.perf_counter()
        response = self.client.chat(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}],
            format=schema,
            options={'temperature': 0},
        )
        return LLMReply(
            text=response.message.content or '',
            input_tokens=int(getattr(response, 'prompt_eval_count', 0) or 0),
            output_tokens=int(getattr(response, 'eval_count', 0) or 0),
            seconds=round(time.perf_counter() - start, 2),
        )

class FakeProvider:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []
        self.calls = 0

    def ask(self, prompt, schema=None):
        self.prompts.append(prompt)
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return LLMReply(
            text=reply,
            input_tokens=len(prompt) // 4,
            output_tokens=len(reply) // 4,
            seconds=0.0,
        )

def demo_replies():
    first = {
        'customer_name': 'Priya Sharma',
        'order_id': '48213',
        'category': 'delivery',
        'urgency': 'high',
        'sentiment': 'negative',
        'amount_usd': 1299.0,
        'summary': 'Laptop has not arrived and the box looks damaged in tracking.',
        'requested_action': 'Deliver by Friday or refund',
    }
    second = dict(first, order_id='ORD-48213')
    return [TICKS + 'json\n' + json.dumps(first) + '\n' + TICKS, json.dumps(second)]

def get_provider(name):
    if name == 'gemini':
        return GeminiProvider()
    if name == 'ollama':
        return OllamaProvider()
    if name == 'fake':
        return FakeProvider(demo_replies())
    raise ValueError('provider must be gemini, ollama or fake')

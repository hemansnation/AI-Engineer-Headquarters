import os
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

SAMPLING = {'temperature', 'top_p', 'top_k'}
COMMON = {'seed', 'max_output_tokens', 'stop_sequences'}

MODELS = {
    'gemma': {'id': 'gemma-4-26b-a4b-it', 'supports': SAMPLING | COMMON, 'thinking': 'minimal'},
    'lite': {'id': 'gemini-3.1-flash-lite', 'supports': SAMPLING | COMMON, 'thinking': None},
    'new': {'id': 'gemini-3.5-flash-lite', 'supports': COMMON, 'thinking': None},
}

@dataclass
class Result:
    text: str
    finish: str
    input_tokens: int
    output_tokens: int
    thought_tokens: int
    seconds: float
    dropped: list = field(default_factory=list)

def make_client():
    key = os.getenv('GEMINI_API_KEY')
    if not key:
        raise ValueError('GEMINI_API_KEY is missing, check your .env file')
    options = types.HttpOptions(
        timeout=30_000,
        retry_options=types.HttpRetryOptions(attempts=3),
    )
    return genai.Client(api_key=key, http_options=options)

def build_config(model_key, settings, system=None, force=False):
    spec = MODELS[model_key]
    kept = {}
    dropped = []
    for name, value in settings.to_dict().items():
        if force or name in spec['supports']:
            kept[name] = value
        else:
            dropped.append(name)
    if system:
        kept['system_instruction'] = system
    if spec['thinking']:
        kept['thinking_config'] = types.ThinkingConfig(thinking_level=spec['thinking'])
    return types.GenerateContentConfig(**kept), dropped

def ask(client, model_key, prompt, settings, system=None, force=False):
    config, dropped = build_config(model_key, settings, system, force)
    start = time.perf_counter()
    response = client.models.generate_content(
        model=MODELS[model_key]['id'],
        contents=prompt,
        config=config,
    )
    seconds = round(time.perf_counter() - start, 2)

    finish = 'NONE'
    if response.candidates and response.candidates[0].finish_reason:
        finish = response.candidates[0].finish_reason.name

    usage = response.usage_metadata
    return Result(
        text=(response.text or '').strip(),
        finish=finish,
        input_tokens=(usage.prompt_token_count or 0) if usage else 0,
        output_tokens=(usage.candidates_token_count or 0) if usage else 0,
        thought_tokens=(usage.thoughts_token_count or 0) if usage else 0,
        seconds=seconds,
        dropped=dropped,
    )
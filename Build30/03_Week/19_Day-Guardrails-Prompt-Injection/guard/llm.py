import re
import time
import ollama
from guard.config import OLLAMA_HOST
_client = None
def get_client():
    global _client
    if _client is None:
        _client = ollama.Client(host=OLLAMA_HOST)
    return _client
def strip_think(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
def chat(model, system, user, max_tokens=300, temperature=0.0):
    start = time.perf_counter()
    response = get_client().chat(
        model=model,
        messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
        options={'temperature': temperature, 'num_predict': max_tokens},
        think=False,
    )
    seconds = round(time.perf_counter() - start, 3)
    return {
        'text': strip_think(response.message.content or ''),
        'seconds': seconds,
        'prompt_tokens': response.prompt_eval_count or 0,
        'output_tokens': response.eval_count or 0,
    }

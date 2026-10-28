import ollama
from .config import SEED
USAGE = {'calls': 0, 'prompt_tokens': 0, 'output_tokens': 0}
def ask(prompt, model, system=None, fmt=None):
    messages = []
    if system:
        messages.append({'role': 'system', 'content': system})
    messages.append({'role': 'user', 'content': prompt})
    resp = ollama.chat(model=model, messages=messages, think=False, format=fmt, options={'temperature': 0, 'seed': SEED})
    USAGE['calls'] += 1
    USAGE['prompt_tokens'] += resp.get('prompt_eval_count') or 0
    USAGE['output_tokens'] += resp.get('eval_count') or 0
    return resp['message']['content'].strip()
def usage_snapshot():
    return dict(USAGE)
def usage_since(before):
    return {k: USAGE[k] - before[k] for k in USAGE}

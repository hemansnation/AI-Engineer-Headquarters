import os
import re
import time
import ollama
from dotenv import load_dotenv
load_dotenv()
MODEL = os.getenv('LLM_MODEL', 'qwen3:4b')
client = ollama.Client(host=os.getenv('OLLAMA_HOST', 'http://localhost:11434'))
THINK_TAGS = re.compile(r'<think>.*?</think>', re.DOTALL)
USAGE = {'calls': 0, 'input': 0, 'output': 0, 'seconds': 0.0}
def chat(messages, stop=None, tools=None):
    options = {'temperature': 0, 'num_ctx': 4096}
    if stop:
        options['stop'] = stop
    kwargs = {'model': MODEL, 'messages': messages, 'options': options}
    if tools:
        kwargs['tools'] = tools
    started = time.perf_counter()
    try:
        response = client.chat(think=False, **kwargs)
    except TypeError:
        response = client.chat(**kwargs)
    seconds = time.perf_counter() - started
    message = response['message']
    result = {
        'text': THINK_TAGS.sub('', message.get('content') or '').strip(),
        'tool_calls': message.get('tool_calls') or [],
        'input_tokens': response.get('prompt_eval_count') or 0,
        'output_tokens': response.get('eval_count') or 0,
        'seconds': round(seconds, 2),
    }
    USAGE['calls'] += 1
    USAGE['input'] += result['input_tokens']
    USAGE['output'] += result['output_tokens']
    USAGE['seconds'] += seconds
    return result

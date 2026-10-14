import os
import ollama
from dotenv import load_dotenv
load_dotenv()
MODEL = os.getenv('OLLAMA_MODEL', 'qwen3:4b')
HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
client = ollama.Client(host=HOST)
def chat(messages, stop=None, temperature=0.0):
    response = client.chat(
        model=MODEL,
        messages=messages,
        think=False,
        options={'temperature': temperature, 'stop': stop or [], 'num_ctx': 8192},
    )
    return {
        'text': response.message.content or '',
        'prompt_tokens': response.prompt_eval_count or 0,
        'completion_tokens': response.eval_count or 0,
    }

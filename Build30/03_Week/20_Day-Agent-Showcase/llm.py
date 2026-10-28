import re
import ollama
from config import OLLAMA_MODEL, OLLAMA_HOST
client = ollama.Client(host=OLLAMA_HOST, timeout=120)
def chat(messages):
    response = client.chat(model=OLLAMA_MODEL, messages=messages, options={'temperature': 0, 'seed': 7})
    text = response['message']['content'] or ''
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    usage = {'prompt': response.get('prompt_eval_count') or 0, 'completion': response.get('eval_count') or 0}
    return text, usage

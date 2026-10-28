import re
import ollama
from config import MODEL, HOST
_client = ollama.Client(host=HOST)
def ask(system, prompt, temperature=0.2, model=None):
    response = _client.chat(
        model=model or MODEL,
        messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': prompt}],
        think=False,
        options={'temperature': temperature},
    )
    text = response.message.content or ''
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    usage = (response.prompt_eval_count or 0, response.eval_count or 0)
    return text, usage

import os
import re
import ollama
from dotenv import load_dotenv
from tracker import track
load_dotenv()
MODEL = os.getenv('OLLAMA_MODEL', 'qwen3:4b')
class LLMUnavailable(Exception):
    pass
def clean(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
@track
def _chat(prompt):
    return ollama.chat(model=MODEL, messages=[{'role': 'user', 'content': prompt}], options={'temperature': 0.2})
def ask(prompt):
    try:
        reply = _chat(prompt)
    except Exception as error:
        raise LLMUnavailable(str(error))
    return clean(reply['message']['content'])

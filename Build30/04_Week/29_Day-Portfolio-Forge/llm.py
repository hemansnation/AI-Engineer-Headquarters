import ollama
from pydantic import BaseModel
from config import OLLAMA_MODEL
from tracker import track_tokens
@track_tokens
def ask(prompt: str, schema: type[BaseModel], system: str = ''):
    messages = []
    if system:
        messages.append({'role': 'system', 'content': system})
    messages.append({'role': 'user', 'content': prompt})
    return ollama.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        format=schema.model_json_schema(),
        options={'temperature': 0, 'num_ctx': 8192},
        think=False,
    )
def ask_structured(prompt: str, schema: type[BaseModel], system: str = '') -> BaseModel:
    response = ask(prompt, schema, system)
    return schema.model_validate_json(response.message.content)

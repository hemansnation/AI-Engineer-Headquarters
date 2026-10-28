from ollama import Client
from pydantic import ValidationError
from config import OLLAMA_HOST, OLLAMA_MODEL
from schemas import Brief
from tracker import track_tokens
client = Client(host=OLLAMA_HOST, timeout=300)
SYSTEM = ('You are a forward deployed engineer reading a discovery call transcript. '
    'Fill the schema using only what the transcript says. '
    'If the call does not say something, use null or unknown. Never guess a number. '
    'Copy numbers exactly as spoken. Return JSON only.')
@track_tokens
def chat(messages):
    return client.chat(model=OLLAMA_MODEL, messages=messages, format=Brief.model_json_schema(), options={'temperature': 0}, think=False)
def draft_brief(transcript, max_tries=3):
    messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': transcript}]
    for attempt in range(max_tries):
        text = chat(messages).message.content
        try:
            return Brief.model_validate_json(text)
        except ValidationError as error:
            messages.append({'role': 'assistant', 'content': text})
            messages.append({'role': 'user', 'content': 'That JSON failed validation. Fix these errors and answer again. ' + str(error)})
    raise RuntimeError('The model did not return a valid brief after ' + str(max_tries) + ' tries. Try a bigger model.')

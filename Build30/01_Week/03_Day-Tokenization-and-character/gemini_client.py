import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

MODEL = os.getenv('GEMINI_MODEL')
_client = None


def get_client():
    global _client
    if _client is None:
        key = os.getenv('GEMINI_API_KEY')
        if not key or not MODEL:
            raise RuntimeError('Set GEMINI_API_KEY and GEMINI_MODEL in your .env file')
        _client = genai.Client(api_key=key)
    return _client


def count_tokens(text):
    result = get_client().models.count_tokens(model=MODEL, contents=text)
    return result.total_tokens


def ask(prompt):
    response = get_client().models.generate_content(model=MODEL, contents=prompt)
    usage = response.usage_metadata
    return {
        'text': response.text,
        'prompt_tokens': usage.prompt_token_count,
        'output_tokens': usage.candidates_token_count,
        'thinking_tokens': getattr(usage, 'thoughts_token_count', None) or 0,
        'total_tokens': usage.total_token_count,
    }
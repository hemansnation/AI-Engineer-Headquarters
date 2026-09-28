import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from tracker import track_tokens

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

client = genai.Client(
    api_key=API_KEY,
    http_options=types.HttpOptions(
        timeout=30_000,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
)

@track_tokens
def ask(prompt, model=MODEL):
    return client.interactions.create(model=model, input=prompt)
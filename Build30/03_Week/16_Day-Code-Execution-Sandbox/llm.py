import os
from dotenv import load_dotenv
from google import genai
load_dotenv()
def ask(prompt: str):
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError('GEMINI_API_KEY is missing in your .env file')
    client = genai.Client(api_key=api_key)
    return client.interactions.create(model=os.getenv('GEMINI_MODEL'), input=prompt)

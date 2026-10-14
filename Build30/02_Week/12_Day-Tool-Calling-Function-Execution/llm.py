import os
from dotenv import load_dotenv
from ollama import Client
load_dotenv()
MODEL = os.getenv('OLLAMA_MODEL', 'qwen3:8b')
HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
client = Client(host=HOST)
def chat(messages, tools):
    return client.chat(
        model=MODEL,
        messages=messages,
        tools=tools,
        think=False,
        options={'temperature': 0},
    )

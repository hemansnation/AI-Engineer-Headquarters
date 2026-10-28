import os
import httpx
def fake_llm(question, docs):
    return f'fake answer to {question} using {len(docs)} documents'
def ollama_llm(question, docs):
    url = os.getenv('OLLAMA_URL', 'http://localhost:11434/api/generate')
    model = os.getenv('OLLAMA_MODEL', 'llama3.2:3b')
    context = '\n'.join(docs)
    prompt = f'Answer only from this context.\n{context}\nQuestion: {question}'
    reply = httpx.post(url, json={'model': model, 'prompt': prompt, 'stream': False}, timeout=120)
    return reply.json()['response']

import re
import ollama
from rag.config import LLM_MODEL, TOP_K

REFUSAL = 'I do not know based on the documents.'
SYSTEM = 'You answer questions using only the context provided. Cite the sources you used like [1] or [2]. If the context does not contain the answer, reply exactly: I do not know based on the documents.'


def build_prompt(question, hits):
    blocks = []
    for number, hit in enumerate(hits, start=1):
        blocks.append(f'[{number}] (source: {hit.chunk.source})\n{hit.chunk.text}')
    context = '\n\n'.join(blocks)
    return f'Context:\n{context}\n\nQuestion: {question}\nAnswer:'


def ask_llm(prompt):
    messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}]
    options = {'temperature': 0}
    try:
        try:
            reply = ollama.chat(model=LLM_MODEL, messages=messages, options=options, think=False)
        except TypeError:
            reply = ollama.chat(model=LLM_MODEL, messages=messages, options=options)
    except ConnectionError:
        raise SystemExit('Ollama is not running. Start it with: ollama serve')
    text = reply['message']['content']
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def answer(retriever, question, mode='hybrid', top_k=TOP_K):
    hits = retriever.search(question, mode=mode, top_k=top_k)
    if not hits:
        return {'answer': REFUSAL, 'hits': []}
    return {'answer': ask_llm(build_prompt(question, hits)), 'hits': hits}

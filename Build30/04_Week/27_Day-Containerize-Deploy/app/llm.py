import httpx

from app.config import settings


def llm_enabled():
    return bool(settings.llm_base_url)


async def ask_llm(question, contexts):
    context_block = '\n'.join(f'- {text}' for text in contexts)
    prompt = (
        'Answer using only the context below. '
        'If the context does not contain the answer, say you do not know.\n\n'
        f'Context:\n{context_block}\n\nQuestion: {question}'
    )
    body = {
        'model': settings.llm_model,
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0,
    }
    headers = {'Authorization': f'Bearer {settings.llm_api_key}'}
    url = settings.llm_base_url.rstrip('/') + '/chat/completions'
    async with httpx.AsyncClient(timeout=settings.llm_timeout) as client:
        response = await client.post(url, json=body, headers=headers)
        response.raise_for_status()
        data = response.json()
    return data['choices'][0]['message']['content']

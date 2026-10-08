from src.config import NO_ANSWER

SYSTEM_RULES = (
    'You answer questions using only the context the user gives you. '
    'Every fact in your answer must come from that context. '
    'After each fact, add the source number in square brackets, like [1]. '
    'If the context does not contain the answer, reply with exactly this sentence: '
    + NO_ANSWER
    + ' Keep the answer short and plain.'
)

def format_context(hits: list[dict]) -> str:
    blocks = []
    for number, hit in enumerate(hits, start=1):
        source = hit['source']
        text = hit['text']
        blocks.append(f'[{number}] source: {source}\n{text}')
    return '\n\n'.join(blocks)

def build_messages(question: str, hits: list[dict]) -> list[dict]:
    context = format_context(hits)
    user_message = f'Context:\n{context}\n\nQuestion: {question}'
    return [
        {'role': 'system', 'content': SYSTEM_RULES},
        {'role': 'user', 'content': user_message},
    ]
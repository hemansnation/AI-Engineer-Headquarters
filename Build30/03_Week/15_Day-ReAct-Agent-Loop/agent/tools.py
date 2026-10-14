import ast
import operator
from pathlib import Path
import numpy as np
TOOLS = {}
NOTES_DIR = Path(__file__).resolve().parent.parent / 'notes'
EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}
_cache = {}
def register_tool(description):
    def decorator(func):
        TOOLS[func.__name__] = {'func': func, 'description': description}
        return func
    return decorator
def safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        left = safe_eval(node.left)
        right = safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError('exponent is too big')
        return OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](safe_eval(node.operand))
    raise ValueError('only numbers and + - * / % ** are allowed')
@register_tool('Do math. Input is one expression like 12*(3+4)')
def calculator(expression):
    return str(safe_eval(ast.parse(expression, mode='eval').body))
@register_tool('Count a letter inside a word. Input is word,letter like strawberry,r')
def count_letter(text):
    word, letter = [part.strip() for part in text.split(',')]
    return str(word.lower().count(letter.lower()))
def get_embedder():
    if 'model' not in _cache:
        from sentence_transformers import SentenceTransformer
        _cache['model'] = SentenceTransformer(EMBED_MODEL)
    return _cache['model']
def load_index():
    if 'vectors' not in _cache:
        chunks = []
        for path in sorted(NOTES_DIR.glob('*.txt')):
            for line in path.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    chunks.append({'source': path.name, 'text': line.strip()})
        texts = [chunk['text'] for chunk in chunks]
        _cache['chunks'] = chunks
        _cache['vectors'] = np.array(get_embedder().encode(texts, normalize_embeddings=True))
    return _cache['chunks'], _cache['vectors']
@register_tool('Search my notes by meaning. Input is a short question')
def search_notes(query):
    chunks, vectors = load_index()
    query_vector = get_embedder().encode([query], normalize_embeddings=True)[0]
    scores = vectors @ query_vector
    best = np.argsort(scores)[::-1][:2]
    lines = []
    for index in best:
        lines.append('[' + chunks[index]['source'] + '] ' + chunks[index]['text'])
    return '\n'.join(lines)

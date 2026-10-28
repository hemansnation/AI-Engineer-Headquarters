from config import EMBED_MODEL, TOP_K_TABLES
from schema_cards import all_cards
_model = None
_cache = {}
def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL)
    return _model
def card_vectors(model):
    if 'cards' not in _cache:
        cards = all_cards()
        _cache['cards'] = (list(cards), model.encode(list(cards.values()), normalize_embeddings=True))
    return _cache['cards']
def pick_tables(question, top_k=TOP_K_TABLES, model=None):
    model = model or get_model()
    names, vectors = card_vectors(model)
    query = model.encode([question], normalize_embeddings=True)[0]
    scores = vectors @ query
    ranked = sorted(zip(names, scores), key=lambda pair: pair[1], reverse=True)
    return [(name, round(float(score), 2)) for name, score in ranked[:top_k]]

import re
from config import EMBED_MODEL
_model = None
def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL)
    return _model
def split_lines(transcript):
    return [line.strip() for line in transcript.splitlines() if line.strip()]
def best_evidence(claims, transcript, threshold, model=None):
    model = model or get_model()
    lines = split_lines(transcript)
    line_vectors = model.encode(lines, normalize_embeddings=True)
    claim_vectors = model.encode(list(claims.values()), normalize_embeddings=True)
    scores = claim_vectors @ line_vectors.T
    results = {}
    for row, name in enumerate(claims):
        index = int(scores[row].argmax())
        score = round(float(scores[row][index]), 2)
        results[name] = {'line': lines[index], 'score': score, 'grounded': score >= threshold}
    return results
def number_in_text(number, text):
    value = str(int(number)) if float(number).is_integer() else str(number)
    return re.search(r'\b' + re.escape(value) + r'\b', text.replace(',', '')) is not None
def check_numbers(brief, transcript):
    values = {'volume_per_month': brief.volume_per_month, 'minutes_per_task': brief.minutes_per_task, 'cost_per_hour_usd': brief.cost_per_hour_usd, 'budget_usd': brief.budget_usd}
    if brief.automatable_share is not None:
        values['automatable_share'] = round(brief.automatable_share * 100)
    return [name for name, value in values.items() if value is not None and not number_in_text(value, transcript)]

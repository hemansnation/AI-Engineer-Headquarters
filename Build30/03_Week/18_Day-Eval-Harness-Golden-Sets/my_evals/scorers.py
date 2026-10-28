from dataclasses import dataclass
from .config import EDIT_PASS, EMBED_PASS
from .textdist import similarity as edit_similarity
PUNCT = ' .,!?;:' + chr(34) + chr(39)
@dataclass
class Score:
    passed: bool
    value: float
    note: str = ''
def normalize(text):
    return ' '.join(text.lower().split()).strip(PUNCT)
def exact(case, output, ctx):
    ok = normalize(output) == normalize(case.expected)
    return Score(ok, 1.0 if ok else 0.0)
def contains(case, output, ctx):
    parts = [p.strip().lower() for p in case.expected.split('|')]
    hits = [p for p in parts if p in output.lower()]
    return Score(len(hits) == len(parts), len(hits) / len(parts), f'missing: {[p for p in parts if p not in hits]}')
def edit(case, output, ctx):
    sim = edit_similarity(normalize(output), normalize(case.expected))
    return Score(sim >= EDIT_PASS, round(sim, 3))
def embed(case, output, ctx):
    sim = ctx['embed'](output, case.expected)
    return Score(sim >= EMBED_PASS, round(sim, 3))
def judge(case, output, ctx):
    passed, reason = ctx['judge'](case, output)
    return Score(passed, 1.0 if passed else 0.0, reason)
SCORERS = {'exact': exact, 'contains': contains, 'edit': edit, 'embed': embed, 'judge': judge}
def score_case(case, output, ctx):
    return SCORERS[case.scorer](case, output, ctx)

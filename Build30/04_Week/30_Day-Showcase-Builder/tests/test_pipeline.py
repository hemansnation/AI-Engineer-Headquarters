import zlib
import numpy as np
from store import EvidenceStore, rrf
from writer import draft_case_study, gather
def fake_embed(texts):
    vectors = np.zeros((len(texts), 64))
    for row, text in enumerate(texts):
        for word in text.lower().split():
            vectors[row, zlib.crc32(word.encode()) % 64] += 1
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)
CHUNKS = [
    {'source': 'day-09', 'text': 'The problem was support tickets taking 3 days to answer because nobody could find the right policy page.'},
    {'source': 'day-09', 'text': 'I chose hybrid search over pure vector search because part numbers like AB-9128 never matched by meaning.'},
    {'source': 'day-09', 'text': 'It broke when chunks cut a sentence in half, so I added overlap and the pass rate moved from 62 to 71 out of 100.'},
    {'source': 'day-22', 'text': 'FastAPI endpoint handled 40 requests per second on my laptop.'},
]
STATS = {'commits': 30, 'first_day': '2026-09-28', 'last_day': '2026-10-28', 'active_days': 30, 'longest_streak': 30}
def test_rrf_rewards_agreement():
    fused = rrf([[1, 2, 3], [1, 3, 2]])
    assert max(fused, key=fused.get) == 1
def test_filter_keeps_one_build():
    store = EvidenceStore(fake_embed)
    store.add_many(CHUNKS)
    hits = store.query('how fast is the endpoint', 2, 'day-22')
    assert {h['source'] for h in hits} == {'day-22'}
def test_save_and_load_roundtrip(tmp_path):
    store = EvidenceStore(fake_embed)
    store.add_many(CHUNKS)
    store.save(tmp_path)
    again = EvidenceStore.load(tmp_path, fake_embed)
    assert again.query('part numbers', 1, 'day-09')[0]['text'] == store.query('part numbers', 1, 'day-09')[0]['text']
def test_verifier_gate_repairs_invented_number():
    store = EvidenceStore(fake_embed)
    store.add_many(CHUNKS)
    evidence = gather(store, 'day-09')
    replies = iter(['Tickets took 3 days and I cut it to 45 minutes.', 'Tickets took 3 days. The pass rate moved from 62 to 71 out of 100.'])
    text, flagged = draft_case_study('day-09', evidence, STATS, lambda prompt: next(replies))
    assert flagged == [] and '45' not in text
def test_gate_masks_when_model_keeps_lying():
    store = EvidenceStore(fake_embed)
    store.add_many(CHUNKS)
    evidence = gather(store, 'day-09')
    text, flagged = draft_case_study('day-09', evidence, STATS, lambda prompt: 'I cut tickets to 45 minutes.')
    assert flagged == ['45'] and 'NUMBER NEEDED' in text

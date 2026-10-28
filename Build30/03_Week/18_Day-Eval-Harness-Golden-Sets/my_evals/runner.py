import json
import time
from datetime import datetime, timezone
from .golden import dataset_hash
from .llm import usage_snapshot, usage_since
from .scorers import score_case
def run_eval(cases, system_fn, ctx, label, golden_path=None):
    started = time.perf_counter()
    before = usage_snapshot()
    results = []
    for c in cases:
        t0 = time.perf_counter()
        try:
            output = system_fn(c.question)
            s = score_case(c, output, ctx)
            passed, value, note = s.passed, s.value, s.note
        except Exception as e:
            output, passed, value, note = '', False, 0.0, f'error: {e}'
        results.append({'id': c.id, 'scorer': c.scorer, 'tags': c.tags, 'must_pass': c.must_pass, 'question': c.question, 'expected': c.expected, 'output': output, 'passed': passed, 'value': value, 'note': note, 'seconds': round(time.perf_counter() - t0, 2)})
    n_pass = sum(r['passed'] for r in results)
    return {
        'label': label,
        'golden_hash': dataset_hash(golden_path) if golden_path else 'none',
        'when': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'total': len(results),
        'passed': n_pass,
        'pass_rate': round(n_pass / len(results), 4) if results else 0.0,
        'seconds': round(time.perf_counter() - started, 1),
        'usage': usage_since(before),
        'results': results,
    }
def save_run(run, folder):
    folder.mkdir(parents=True, exist_ok=True)
    name = run['label'] + '.json'
    path = folder / name
    path.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding='utf-8')
    return path
def load_run(folder, label):
    return json.loads((folder / f'{label}.json').read_text(encoding='utf-8'))

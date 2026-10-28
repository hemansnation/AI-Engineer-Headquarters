import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from agent import run_agent
from config import CANARY, FUZZY_THRESHOLD, MIN_PASS_RATE, MAX_REGRESSION, OLLAMA_MODEL
from embeddings import embed, cosine
from guardrail import mask_pii_entities
from evals.golden_set import CASES
EVAL_LOG = Path('logs/evals.jsonl')
EVAL_RUNS = Path('logs/eval_runs.jsonl')
def passes(check, record):
    kind, final = check['type'], record['final']
    if kind == 'contains':
        return check['value'].lower() in final.lower()
    if kind == 'not_contains':
        return check['value'].lower() not in final.lower()
    if kind == 'status':
        return record['status'] == check['value']
    if kind == 'flag':
        return check['value'] in record['flags']
    if kind == 'no_pii':
        return not mask_pii_entities(final)[1]
    if kind == 'no_leak':
        return CANARY.lower() not in final.lower()
    if kind == 'fuzzy':
        a, b = embed([final, check['value']])
        return cosine(a, b) >= FUZZY_THRESHOLD
    return False
def previous_entry():
    if not EVAL_LOG.exists():
        return None
    lines = [line for line in EVAL_LOG.read_text(encoding='utf-8').splitlines() if line.strip()]
    return json.loads(lines[-1]) if lines else None
def main():
    results = {}
    for case in CASES:
        record = run_agent(case['task'], ledger=EVAL_RUNS)
        results[case['id']] = all(passes(c, record) for c in case['checks'])
        print(('PASS ' if results[case['id']] else 'FAIL ') + case['id'] + ' (' + record['status'] + ')')
    total, passed = len(CASES), sum(results.values())
    security = [c['id'] for c in CASES if c['category'] == 'security']
    security_failed = [i for i in security if not results[i]]
    rate = passed / total
    previous = previous_entry()
    regressed = [i for i, ok in results.items() if previous and previous['results'].get(i) and not ok]
    gate_reasons = []
    if security_failed:
        gate_reasons.append('security cases failed: ' + ', '.join(security_failed))
    if rate < MIN_PASS_RATE:
        gate_reasons.append('pass rate ' + format(rate, '.0%') + ' is below the minimum ' + format(MIN_PASS_RATE, '.0%'))
    if previous and rate < previous['pass_rate'] - MAX_REGRESSION:
        gate_reasons.append('pass rate dropped from ' + format(previous['pass_rate'], '.0%') + ' to ' + format(rate, '.0%'))
    entry = {
        'ts': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'model': OLLAMA_MODEL,
        'total': total,
        'passed': passed,
        'pass_rate': round(rate, 4),
        'security_total': len(security),
        'security_passed': len(security) - len(security_failed),
        'regressed': regressed,
        'results': results,
        'gate': 'pass' if not gate_reasons else 'fail',
        'gate_reasons': gate_reasons,
    }
    EVAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with EVAL_LOG.open('a', encoding='utf-8') as f:
        f.write(json.dumps(entry, sort_keys=True) + '\n')
    print('score: ' + str(passed) + '/' + str(total) + ' security: ' + str(entry['security_passed']) + '/' + str(len(security)))
    if previous:
        print('before: ' + str(previous['passed']) + '/' + str(previous['total']) + ' after: ' + str(passed) + '/' + str(total))
    if regressed:
        print('regressed cases: ' + ', '.join(regressed))
    print('gate: ' + entry['gate'] + ' ' + '; '.join(gate_reasons))
    sys.exit(0 if not gate_reasons else 1)
if __name__ == '__main__':
    main()

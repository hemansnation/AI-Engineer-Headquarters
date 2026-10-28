import heapq
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from recorder import load_runs, verify_chain
def percentile(values, p):
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]
def last_two_evals(path='logs/evals.jsonl'):
    path = Path(path)
    if not path.exists():
        return []
    lines = [line for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    return [json.loads(line) for line in lines[-2:]]
def build_report():
    runs = load_runs()
    intact, bad_line, count = verify_chain()
    statuses = Counter(r['status'] for r in runs)
    latencies = [r['latency_s'] for r in runs]
    slowest = heapq.nlargest(3, runs, key=lambda r: r['latency_s'])
    failures = [r for r in runs if r['status'] != 'ok']
    flags = Counter(flag for r in runs for flag in r['flags'])
    lines = ['# Agent showcase', 'Generated ' + datetime.now(timezone.utc).isoformat(timespec='seconds') + ' from logs/runs.jsonl', '']
    lines.append('## Ledger')
    lines.append('Runs recorded: ' + str(count))
    lines.append('Hash chain intact: ' + ('yes' if intact else 'NO, first bad line is ' + str(bad_line)))
    lines += ['', '## Run outcomes']
    for status, n in statuses.most_common():
        lines.append('- ' + status + ': ' + str(n))
    lines += ['', '## Latency']
    lines.append('p50: ' + str(percentile(latencies, 50)) + ' s, p95: ' + str(percentile(latencies, 95)) + ' s')
    for r in slowest:
        lines.append('- ' + r['run_id'] + ' took ' + str(r['latency_s']) + ' s: ' + r['task'][:60])
    lines += ['', '## Guardrail activity']
    for flag, n in flags.most_common():
        lines.append('- ' + flag + ': ' + str(n))
    lines += ['', '## Every run that did not end ok, unedited']
    for r in failures:
        lines.append('- ' + r['run_id'] + ' ' + r['status'] + ' | ' + r['task'][:60] + ' | ' + r['final'][:80])
    evals = last_two_evals()
    lines += ['', '## Eval score']
    if evals:
        latest = evals[-1]
        lines.append('Latest: ' + str(latest['passed']) + '/' + str(latest['total']) + ' on ' + latest['model'] + ', gate ' + latest['gate'])
        lines.append('Security cases: ' + str(latest['security_passed']) + '/' + str(latest['security_total']))
        if len(evals) == 2:
            lines.append('Before: ' + str(evals[0]['passed']) + '/' + str(evals[0]['total']) + ' After: ' + str(latest['passed']) + '/' + str(latest['total']))
        failed = [i for i, ok in latest['results'].items() if not ok]
        lines.append('Failed cases: ' + (', '.join(failed) if failed else 'none'))
    else:
        lines.append('No eval run found. Run python -m evals.run_evals first.')
    lines += ['', '## What this agent cannot stop', 'Read SECURITY.md, it lists the assumptions and the known gaps.']
    return '\n'.join(lines) + '\n'
if __name__ == '__main__':
    text = build_report()
    Path('SHOWCASE.md').write_text(text, encoding='utf-8')
    print(text)

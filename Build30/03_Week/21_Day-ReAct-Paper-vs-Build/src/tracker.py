import json
from datetime import datetime, timezone
from pathlib import Path
LOG = Path(__file__).resolve().parent.parent / 'out' / 'token_log.jsonl'
def record(run_id, step, usage):
    LOG.parent.mkdir(exist_ok=True)
    row = {
        'time': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'run': run_id,
        'step': step,
        'input_tokens': usage['input_tokens'],
        'output_tokens': usage['output_tokens'],
        'seconds': usage['seconds'],
    }
    with LOG.open('a', encoding='utf-8') as file:
        file.write(json.dumps(row) + '\n')
def totals():
    result = {'calls': 0, 'input_tokens': 0, 'output_tokens': 0}
    if not LOG.exists():
        return result
    for line in LOG.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        result['calls'] += 1
        result['input_tokens'] += row['input_tokens']
        result['output_tokens'] += row['output_tokens']
    return result

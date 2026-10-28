import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
LEDGER = Path('logs/runs.jsonl')
GENESIS = '0' * 64
def digest(body):
    payload = json.dumps(body, sort_keys=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()
def load_runs(path=LEDGER):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]
def append_run(record, path=LEDGER):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    runs = load_runs(path)
    body = dict(record)
    body['run_id'] = uuid.uuid4().hex[:8]
    body['ts'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    body['prev_hash'] = runs[-1]['hash'] if runs else GENESIS
    body['hash'] = digest(body)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(body, sort_keys=True) + '\n')
    return body
def verify_chain(path=LEDGER):
    runs = load_runs(path)
    previous = GENESIS
    for number, run in enumerate(runs, start=1):
        claimed = run.get('hash')
        body = {k: v for k, v in run.items() if k != 'hash'}
        if body.get('prev_hash') != previous or digest(body) != claimed:
            return False, number, len(runs)
        previous = claimed
    return True, None, len(runs)

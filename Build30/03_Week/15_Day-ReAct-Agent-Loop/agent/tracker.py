import json
from datetime import datetime, timezone
from pathlib import Path
class TokenTracker:
    def __init__(self):
        self.rows = []
    def record(self, step, prompt_tokens, completion_tokens):
        self.rows.append({'step': step, 'prompt': prompt_tokens, 'completion': completion_tokens})
    def total(self):
        return sum(row['prompt'] + row['completion'] for row in self.rows)
    def table(self):
        lines = ['step  prompt  completion']
        for row in self.rows:
            lines.append(str(row['step']).ljust(6) + str(row['prompt']).ljust(8) + str(row['completion']))
        lines.append('total tokens: ' + str(self.total()))
        return '\n'.join(lines)
    def save(self, path, question, status):
        record = {
            'time': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'question': question,
            'status': status,
            'rows': self.rows,
            'total': self.total(),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as file:
            file.write(json.dumps(record) + '\n')

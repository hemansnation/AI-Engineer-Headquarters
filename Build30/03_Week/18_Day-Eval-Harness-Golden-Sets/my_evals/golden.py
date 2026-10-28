import json
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
@dataclass
class Case:
    id: str
    question: str
    expected: str
    scorer: str
    tags: list = field(default_factory=list)
    source: str = 'hand_written'
    must_pass: bool = False
def validate(cases, scorer_names):
    seen = set()
    for c in cases:
        if c.id in seen:
            raise ValueError(f'duplicate id {c.id}')
        seen.add(c.id)
        if c.scorer not in scorer_names:
            raise ValueError(f'{c.id}: unknown scorer {c.scorer}')
        if not c.question.strip() or not c.expected.strip():
            raise ValueError(f'{c.id}: question and expected cannot be empty')
    return cases
def load_golden(path, scorer_names):
    cases = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.strip():
            cases.append(Case(**json.loads(line)))
    return validate(cases, scorer_names)
def dataset_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
def save_golden(path, cases):
    lines = [json.dumps(asdict(c), ensure_ascii=False) for c in cases]
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')
def add_case(path, case, scorer_names):
    cases = load_golden(path, scorer_names) if Path(path).exists() else []
    cases.append(case)
    validate(cases, scorer_names)
    save_golden(path, cases)

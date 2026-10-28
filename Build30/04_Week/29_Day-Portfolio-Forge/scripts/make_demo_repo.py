import os
import subprocess
import sys
from pathlib import Path

PY_RUNNER = '''def score_case(expected, actual, mode):
    if not actual.strip():
        return 0.0
    if mode == 'exact':
        return 1.0 if expected.strip() == actual.strip() else 0.0
    words = set(expected.lower().split())
    hits = sum(1 for word in words if word in actual.lower())
    return hits / max(len(words), 1)
def run_all(cases, mode):
    scores = [score_case(c['expected'], c['actual'], mode) for c in cases]
    return sum(scores) / max(len(scores), 1)
'''
PY_SMALL = '''def helper(value):
    return value
def pipeline(items):
    return [helper(item) for item in items]
'''
README_18 = '''# Day 18 Eval harness and golden sets
I built the golden set from real failures instead of synthetic examples because synthetic cases passed every time.
I chose exact match for ids and an LLM judge only for free text, because a judge on everything was 4x slower.
The first version broke when the judge passed empty answers, and I fixed it by checking length first.
Pass rate went from 62% to 71% after the reranker change.
The limitation is that 40 cases is a small set.
'''
RECEIPTS_18 = '''[[receipt]]
label = 'pass rate before reranker'
value = 62
unit = '%'
[[receipt]]
label = 'pass rate after reranker'
value = 71
unit = '%'
[[receipt]]
label = 'golden set cases'
value = 40
unit = ''
'''
def readme(title: str, line: str) -> str:
    return f'# Day {title}\n{line}\n'
DAYS = [
    (1, 'Day 01 First LLM call', readme('01 First LLM call', 'I logged every token to a file.'), {'tracker.py': PY_SMALL}, ['2026-09-28T10:00:00+05:30', '2026-09-28T10:40:00+05:30', '2026-09-29T09:00:00+05:30'], ['add token tracker', 'wip', 'fix timeout']),
    (3, 'Day 03 Tokenization', readme('03 Tokenization', 'Hindi cost 40 tokens for one sentence.'), {'bpe.py': PY_SMALL}, ['2026-09-30T18:00:00+05:30', '2026-09-30T18:30:00+05:30'], ['add character tokenizer', 'wip']),
    (5, 'Day 05 Vector store', readme('05 Vector store', 'A list of records with a vector and metadata.'), {'store.py': PY_SMALL}, ['2026-10-02T11:00:00+05:30', '2026-10-02T15:00:00+05:30'], ['add vector store', 'save to disk']),
    (12, 'Day 12 Tool calling', readme('12 Tool calling', 'I registered tools with a decorator.'), {'tools.py': PY_SMALL}, ['2026-10-09T10:00:00+05:30', '2026-10-09T13:00:00+05:30', '2026-10-10T10:00:00+05:30'], ['add tool registry', 'block bad params', 'wip']),
    (18, 'Day 18 Eval harness and golden sets', README_18, {'eval_runner.py': PY_RUNNER, 'test_eval_runner.py': PY_SMALL, 'demo.png': 'png', 'receipts.toml': RECEIPTS_18}, ['2026-10-14T10:00:00+05:30', '2026-10-14T10:40:00+05:30', '2026-10-14T11:10:00+05:30', '2026-10-14T16:00:00+05:30', '2026-10-14T16:30:00+05:30', '2026-10-15T09:00:00+05:30', '2026-10-15T09:20:00+05:30'], ['add golden set from real failures', 'switch to exact match for ids because the judge was slow', 'fix judge passing empty answers', 'add reranker change and rerun evals', 'wip', 'add ci gate', 'write readme']),
    (20, 'Day 20 Agent showcase', readme('20 Agent showcase', 'I showed the failed runs too.'), {'agent.py': PY_SMALL}, ['2026-10-16T10:00:00+05:30', '2026-10-17T10:00:00+05:30'], ['add agent loop', 'wip']),
]
def git(repo: Path, *args: str, when: str | None = None) -> None:
    env = dict(os.environ)
    if when:
        env['GIT_AUTHOR_DATE'] = when
        env['GIT_COMMITTER_DATE'] = when
    subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True, env=env)
def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if content == 'png':
        path.write_bytes(b'\x89PNG\r\n')
    else:
        path.write_text(content, encoding='utf-8')
def make_demo_repo(target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    git(target, 'init', '-q')
    git(target, 'config', 'user.name', 'Demo Builder')
    git(target, 'config', 'user.email', 'demo@example.com')
    write(target / 'DEMO_DATA.txt', 'practice repo, not real work\n')
    for day, folder_title, text, files, dates, subjects in DAYS:
        folder = target / 'week-x' / f'day-{day:02d}'
        write(folder / 'README.md', text)
        for step, (when, subject) in enumerate(zip(dates, subjects)):
            for name, content in files.items():
                write(folder / name, content + f'\nstep = {step}\n' if name.endswith('.py') else content)
            git(target, 'add', '-A')
            git(target, 'commit', '-q', '-m', subject, when=when)
    return target
if __name__ == '__main__':
    print(make_demo_repo(Path(sys.argv[1] if len(sys.argv) > 1 else 'demo_repo')))

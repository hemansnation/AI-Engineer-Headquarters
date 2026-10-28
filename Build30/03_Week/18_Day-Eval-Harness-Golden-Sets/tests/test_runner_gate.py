import pytest
from my_evals.golden import Case, save_golden
from my_evals.runner import run_eval, save_run, load_run
from my_evals.report import compare_runs, format_run
from my_evals.gate import check_gate
CASES = [Case('a', 'qa', '30 days', 'contains', ['r'], must_pass=True), Case('b', 'qb', '49', 'contains', ['s']), Case('c', 'qc', 'x', 'contains', ['s'])]
def system(answers):
    def run(q):
        if answers.get(q) == 'boom':
            raise RuntimeError('model down')
        return answers.get(q, '')
    return run
def make(tmp_path, label, answers):
    p = tmp_path / 'g.jsonl'
    save_golden(p, CASES)
    return run_eval(CASES, system(answers), {}, label, p)
def test_run_scores_and_catches_errors(tmp_path):
    run = make(tmp_path, 'v1', {'qa': 'You have 30 days', 'qb': 'boom'})
    assert run['passed'] == 1 and run['total'] == 3
    assert 'error' in [r for r in run['results'] if r['id'] == 'b'][0]['note']
    assert 'FAIL b' in format_run(run)
def test_compare_fixed_and_broke(tmp_path):
    base = make(tmp_path, 'v1', {'qa': '30 days', 'qb': 'nope'})
    new = make(tmp_path, 'v2', {'qa': 'nope', 'qb': '49 rupees'})
    d = compare_runs(base, new)
    assert d['fixed'] == ['b'] and d['broke'] == ['a'] and d['broke_must_pass'] == ['a']
def test_gate_blocks_must_pass_break(tmp_path):
    base = make(tmp_path, 'v1', {'qa': '30 days', 'qb': 'nope'})
    new = make(tmp_path, 'v2', {'qa': 'nope', 'qb': '49', 'qc': 'x'})
    ok, reasons = check_gate(base, new)
    assert not ok and 'must-pass' in reasons[0]
def test_gate_allows_improvement(tmp_path):
    base = make(tmp_path, 'v1', {'qa': '30 days'})
    new = make(tmp_path, 'v2', {'qa': '30 days', 'qb': '49'})
    assert check_gate(base, new)[0]
def test_gate_blocks_big_drop(tmp_path):
    base = make(tmp_path, 'v1', {'qa': '30 days', 'qb': '49', 'qc': 'x'})
    new = make(tmp_path, 'v2', {'qa': '30 days', 'qb': '49'})
    ok, reasons = check_gate(base, new)
    assert not ok and 'dropped' in reasons[0]
def test_compare_rejects_different_golden(tmp_path):
    base = make(tmp_path, 'v1', {})
    new = dict(base, golden_hash='other')
    with pytest.raises(ValueError):
        compare_runs(base, new)
def test_save_and_load(tmp_path):
    run = make(tmp_path, 'v1', {'qa': '30 days'})
    save_run(run, tmp_path / 'runs')
    assert load_run(tmp_path / 'runs', 'v1')['passed'] == run['passed']

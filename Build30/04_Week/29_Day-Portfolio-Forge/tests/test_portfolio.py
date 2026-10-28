import argparse
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pytest
import embed
import tracker
from assembler import render_case_study
from audit import audit, split_receipts
from drafter import draft
from gitlog import commits_by_day, day_of, decision_subjects, read_commits
from main import run
from make_demo_repo import make_demo_repo
from ranker import rank, rank_all, rescale
from receipts import build_receipts, extract_numbers, failures, verify
from schemas import CaseStudy, Commit, Receipt
from sessions import longest_streak, make_sessions
from signals import find_day_dirs, scan_all
from skim import skim
from snippet import best_function, pick_snippet
@pytest.fixture(scope='session')
def repo(tmp_path_factory):
    return make_demo_repo(tmp_path_factory.mktemp('demo') / 'repo')
@pytest.fixture(scope='session')
def commits(repo):
    return read_commits(repo)
@pytest.fixture(scope='session')
def builds(repo, commits):
    return scan_all(repo, commits_by_day(commits))
def commit_at(when: str) -> Commit:
    return Commit(sha='x', author='a', when=datetime.fromisoformat(when), subject='s')
def study(result: str) -> CaseStudy:
    return CaseStudy(headline='Evals that catch real failures', problem='My agent failed in ways I could not see.', decision='I built a golden set from real failures.', tradeoff='It costs more time to collect.', result=result, what_broke='The judge passed empty answers.')
def test_read_commits_sorted(commits):
    assert len(commits) == 19
    assert commits[0].subject == 'add token tracker'
    assert [c.when for c in commits] == sorted(c.when for c in commits)
def test_day_of_paths():
    assert day_of('week-x/day-01/a.py') == 1
    assert day_of('day-10/x') == 10
    assert day_of('Day_07/x') == 7
    assert day_of('src/utils.py') is None
def test_commits_by_day(commits):
    assert set(commits_by_day(commits)) == {1, 3, 5, 12, 18, 20}
def test_decision_subjects_skip_noise(commits):
    picked = decision_subjects(commits_by_day(commits)[18], 18)
    assert picked and 'wip' not in picked
    assert all(' ' in subject for subject in picked)
def test_sessions_merge_by_gap():
    items = [commit_at('2026-10-01T10:00:00'), commit_at('2026-10-01T10:30:00'), commit_at('2026-10-01T12:00:00'), commit_at('2026-10-01T16:00:00')]
    sessions = make_sessions(items, 90)
    assert [s.commits for s in sessions] == [3, 1]
    assert sessions[0].minutes == 120
def test_sessions_unsorted_input():
    items = [commit_at('2026-10-01T16:00:00'), commit_at('2026-10-01T10:00:00')]
    assert len(make_sessions(items, 90)) == 2
def test_longest_streak():
    days = ['2026-10-01', '2026-10-02', '2026-10-03', '2026-10-05', '2026-10-06']
    assert longest_streak([commit_at(d + 'T10:00:00') for d in days]) == 3
def test_titles_drop_the_day_prefix(builds):
    assert next(b for b in builds if b.day == 18).title == 'Eval harness and golden sets'
def test_nested_repos_are_not_scanned(tmp_path):
    outer = make_demo_repo(tmp_path / 'outer')
    make_demo_repo(outer / 'day-29' / 'demo_repo')
    found = find_day_dirs(outer)
    assert 'demo_repo' not in str(found[18])
    assert 29 in found and len(found) == 7
def test_signals_find_strongest(builds):
    day18 = next(b for b in builds if b.day == 18)
    assert len(builds) == 6
    assert day18.has_tests and day18.has_demo
    assert (day18.decisions, day18.measured_results, day18.failure_notes) == (4, 3, 2)
def test_rank_evidence_beats_fit(builds):
    assert rank_all(builds)[0].day == 18
    skewed = [1.0 if b.day == 20 else 0.0 for b in builds]
    assert rank_all(builds, skewed)[0].day == 18
    assert [s.day for s in rank(builds, k=2)] == [s.day for s in rank_all(builds)[:2]]
def test_rescale():
    assert rescale([1, 2, 3]) == [0.0, 0.5, 1.0]
    assert rescale([4, 4]) == [0.5, 0.5]
def test_extract_numbers():
    text = 'Pass rate went from 62% to 71%, 4x faster, 12k tokens, 1,200 lines in 6.4 hours with qwen3.5:4b in 2026'
    found = [(value, unit) for _, value, _, unit in extract_numbers(text)]
    assert found == [(62.0, '%'), (71.0, '%'), (4.0, 'x'), (12000.0, ''), (1200.0, 'line'), (6.4, 'h')]
def test_verify_statuses():
    receipts = [Receipt(label='before', value=62, unit='%'), Receipt(label='hours', value=6.4, unit='h'), Receipt(label='commits', value=7, unit='commit')]
    status = {f.raw: f.status for f in verify('62% and 6.4 hours and 6 hours and 5 hours and 62 ms and 7 commits and 9 commits', receipts)}
    assert status['62%'] == 'supported'
    assert status['6.4 hours'] == 'supported'
    assert status['6 hours'] == 'rounded'
    assert status['5 hours'] == 'unsupported'
    assert status['62 ms'] == 'unit_mismatch'
    assert status['7 commits'] == 'supported'
    assert status['9 commits'] == 'unsupported'
    assert {f.raw for f in failures(verify('62 ms and 9 commits and 62%', receipts))} == {'62 ms', '9 commits'}
def test_receipts_include_toml(builds, commits):
    day18 = next(b for b in builds if b.day == 18)
    receipts = build_receipts(day18, commits_by_day(commits)[18])
    labels = {r.label: r for r in receipts}
    assert labels['pass rate after reranker'].value == 71
    assert labels['commits'].value == 7
    assert labels['pass rate after reranker'].source == 'receipts.toml'
def test_snippet_picks_branchy_function(repo, commits):
    found = pick_snippet(repo, commits_by_day(commits)[18], 18)
    assert found.name == 'score_case'
    assert best_function('def main():\n    return 1\n    return 2\n') is None
def draft_inputs(builds, commits):
    day18 = next(b for b in builds if b.day == 18)
    return day18, build_receipts(day18, commits_by_day(commits)[18]), ['switch to exact match because the judge was slow']
def test_draft_accepts_grounded_numbers(builds, commits):
    build, receipts, decisions = draft_inputs(builds, commits)
    result = draft(build, 'AI engineer', receipts, decisions, lambda p, s, sys: study('Pass rate went from 62% to 71%.'))
    assert (result.source, result.attempts) == ('model', 1)
def test_draft_retries_with_feedback_then_succeeds(builds, commits):
    build, receipts, decisions = draft_inputs(builds, commits)
    prompts = []
    def llm(prompt, schema, system):
        prompts.append(prompt)
        return study('Pass rate went from 62% to 99%.' if len(prompts) == 1 else 'Pass rate went from 62% to 71%.')
    result = draft(build, 'AI engineer', receipts, decisions, llm)
    assert result.attempts == 2 and result.source == 'model'
    assert '99%' in prompts[1]
def test_draft_falls_back_when_numbers_stay_invented(builds, commits):
    build, receipts, decisions = draft_inputs(builds, commits)
    result = draft(build, 'AI engineer', receipts, decisions, lambda p, s, sys: study('It improved 99%.'))
    assert (result.source, result.attempts) == ('template', 3)
    assert '99%' in result.error
def test_draft_falls_back_when_model_is_down(builds, commits):
    build, receipts, decisions = draft_inputs(builds, commits)
    def down(prompt, schema, system):
        raise ConnectionError('ollama is not running')
    result = draft(build, 'AI engineer', receipts, decisions, down)
    assert result.source == 'template' and result.error.startswith('ConnectionError')
def test_skim_passes_a_tight_case_study():
    text = render_case_study(study('Pass rate went from 62% to 71%.'), 'Asha', 18, [Receipt(label='commits', value=7, unit='commit')], None)
    report = skim(text)
    assert report.passed, report.issues
def test_skim_ignores_receipt_lists_and_technical_nouns():
    text = '# Eval harness that works\n\n## Problem\n\n3 bugs\n\n## Decision\n\nx\n\n## Tradeoff\n\ny\n\n## Result\n\nz\n\n## Receipts\n\n' + '\n'.join(['- one two three four five six seven eight nine ten'] * 10) + '\n'
    assert skim(text).passed
def test_skim_flags_problems():
    long_text = '# Title\n\n## Problem\n\n' + ' '.join(['word'] * 400) + '\n\n## Decision\n\nrobust [fill in: this]\n'
    code = '```python\n' + '\n'.join(['x = 1'] * 20) + '\n```\n'
    issues = ' | '.join(skim(long_text + code).issues)
    for expected in ('reads in', 'code block has 20 lines', 'no number', 'longer than 70', 'missing section: tradeoff', 'banned words: robust', '1 placeholders'):
        assert expected in issues
def test_tracker_logs_tokens(tmp_path, monkeypatch):
    log = tmp_path / 'log.jsonl'
    monkeypatch.setattr(tracker, 'LOG_FILE', log)
    class Reply:
        prompt_eval_count = 10
        eval_count = 5
    tracker.track_tokens(lambda: Reply())()
    assert tracker.running_totals(log)['tokens'] == 15
def test_role_fit_is_a_dot_product(monkeypatch):
    monkeypatch.setattr(embed, 'encode', lambda texts: np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]], dtype=np.float32))
    assert embed.role_fit('role', ['a', 'b']) == pytest.approx([0.6, 0.0])
def make_args(repo, out, **extra):
    base = dict(repo=str(repo), role='AI engineer', author='Asha', lead=None, offline=False, out=str(out))
    return argparse.Namespace(**{**base, **extra})
def test_end_to_end_offline(repo, tmp_path):
    summary = run(make_args(repo, tmp_path, offline=True))
    assert summary['lead'].day == 18
    text = (tmp_path / 'CASE_STUDY.md').read_text()
    assert text.startswith('DEMO DATA') and 'score_case' in text
    assert (tmp_path / 'PORTFOLIO_README.md').exists() and (tmp_path / 'score_report.json').exists()
def test_end_to_end_with_fakes_and_lead_override(repo, tmp_path):
    good = lambda p, s, sys: study('Pass rate went from 62% to 71%.')
    summary = run(make_args(repo, tmp_path), llm=good, fit_fn=lambda role, texts: [0.1 * i for i in range(len(texts))])
    assert summary['source'] == 'model, 1 attempt(s)'
    assert run(make_args(repo, tmp_path, lead=5), llm=good, fit_fn=lambda role, texts: [0.0] * len(texts))['lead'].day == 5
def test_audit_reads_receipts_back_from_the_file(repo, tmp_path):
    run(make_args(repo, tmp_path, offline=True))
    body, receipts = split_receipts((tmp_path / 'CASE_STUDY.md').read_text())
    assert {r.label: r.value for r in receipts}['pass rate after reranker'] == 71
    assert 'return 1.0 if expected' not in body
def test_audit_catches_a_number_added_by_hand(repo, tmp_path):
    run(make_args(repo, tmp_path, offline=True))
    path = tmp_path / 'CASE_STUDY.md'
    original = path.read_text()
    assert not [i for i in audit(original) if 'is unsupported' in i or 'is unit_mismatch' in i]
    edited = original.replace('[fill in: your measured result]', 'Latency dropped by 48%.')
    assert '48% is unsupported' in audit(edited)
    fixed = edited.replace('48%', '71%')
    assert not [i for i in audit(fixed) if 'is unsupported' in i]

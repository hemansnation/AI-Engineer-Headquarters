import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agents import Coder, Reviewer
from coordinator import Coordinator
from parsing import FENCE, extract_code, parse_verdict
GOOD = 'def add(a: int, b: int) -> int:\n    ' + chr(39) * 3 + 'Add two numbers.' + chr(39) * 3 + '\n    return a + b'
NO_HINTS = 'def add(a, b):\n    return a + b'
def wrap(code):
    return FENCE + 'python\n' + code + '\n' + FENCE
def scripted(replies):
    queue = list(replies)
    calls = []
    def fake(system, prompt, temperature=0.2):
        calls.append(prompt)
        return queue.pop(0), (100, 50)
    fake.calls = calls
    return fake
def make(coder_replies, reviewer_replies, **options):
    coder_ask = scripted(coder_replies)
    reviewer_ask = scripted(reviewer_replies)
    coordinator = Coordinator(Coder(coder_ask), Reviewer(reviewer_ask), **options)
    return coordinator, coder_ask, reviewer_ask
def test_approved_on_round_two():
    coordinator, coder_ask, _ = make(
        [wrap(GOOD), wrap(GOOD + '\n# fixed')],
        ['VERDICT: REVISE\n- handle None input', 'VERDICT: APPROVED'],
    )
    result = coordinator.run('add numbers')
    assert result.approved and result.stop_reason == 'approved'
    assert len(result.history.items) == 2
    assert 'handle None input' in coder_ask.calls[1]
def test_cap_at_three_ships_last_draft():
    drafts = [wrap(GOOD + '\n# v' + str(i)) for i in range(3)]
    coordinator, _, _ = make(drafts, ['VERDICT: REVISE\n- still wrong'] * 3)
    result = coordinator.run('add numbers')
    assert not result.approved and result.stop_reason == 'cap'
    assert len(result.history.items) == 3
    assert result.code.endswith('# v2')
def test_static_gate_skips_reviewer_call():
    coordinator, _, reviewer_ask = make([wrap(NO_HINTS), wrap(GOOD)], ['VERDICT: APPROVED'])
    result = coordinator.run('add numbers')
    assert result.approved
    assert len(reviewer_ask.calls) == 1
    assert result.history.items[0].verdict == 'REVISE'
    assert 'type hints' in result.history.items[0].feedback
def test_identical_draft_stops_as_stalled():
    coordinator, _, _ = make([wrap(GOOD), wrap(GOOD)], ['VERDICT: REVISE\n- make it better'] * 2)
    result = coordinator.run('add numbers')
    assert result.stop_reason == 'stalled'
    assert len(result.history.items) == 2
def test_embedding_stall_uses_cosine():
    vectors = {GOOD: [1.0, 0.0], GOOD + '\n# v1': [0.999, 0.01]}
    coordinator, _, _ = make(
        [wrap(GOOD), wrap(GOOD + '\n# v1')],
        ['VERDICT: REVISE\n- x'] * 2,
        embed=lambda text: vectors[text],
    )
    assert coordinator.run('add numbers').stop_reason == 'stalled'
def test_budget_stops_before_a_round_that_cannot_fit():
    drafts = [wrap(GOOD + '\n# v' + str(i)) for i in range(3)]
    coordinator, _, _ = make(drafts, ['VERDICT: REVISE\n- x'] * 3, token_budget=700)
    result = coordinator.run('add numbers')
    assert result.stop_reason == 'budget'
    assert len(result.history.items) == 2
def test_not_approved_is_not_approved():
    assert parse_verdict('VERDICT: NOT APPROVED')[0] == 'REVISE'
    assert parse_verdict('VERDICT: REVISE\n- do not say APPROVED yet')[0] == 'REVISE'
    assert parse_verdict('verdict: approved')[0] == 'APPROVED'
    assert parse_verdict('**VERDICT: APPROVED**')[0] == 'APPROVED'
    assert parse_verdict('Looks good to me')[0] == 'REVISE'
def test_extract_code_with_and_without_fence():
    assert extract_code(wrap('x = 1')) == 'x = 1'
    assert extract_code('x = 2') == 'x = 2'

from my_evals.golden import Case
from my_evals.scorers import score_case, normalize
CTX = {'embed': lambda a, b: 0.9, 'judge': lambda c, o: ('does not' in o, 'fake')}
def test_normalize():
    assert normalize('  30  Days. ') == '30 days'
def test_exact():
    assert score_case(Case('a', 'q', '30 days', 'exact'), '30 days.', CTX).passed
    assert not score_case(Case('a', 'q', '30 days', 'exact'), 'It is 30 days.', CTX).passed
def test_contains_all_parts():
    c = Case('a', 'q', 'saturday|9', 'contains')
    assert score_case(c, 'Open Monday to Saturday from 9 AM', CTX).passed
    s = score_case(c, 'Open on Saturday', CTX)
    assert not s.passed and s.value == 0.5
def test_edit_threshold():
    c = Case('a', 'q', 'Shipping costs 49 rupees.', 'edit')
    assert score_case(c, 'shipping costs 49 rupees', CTX).passed
    assert not score_case(c, 'Shipping is free for you.', CTX).passed
def test_embed_and_judge_use_ctx():
    assert score_case(Case('a', 'q', 'x', 'embed'), 'y', CTX).passed
    assert score_case(Case('a', 'q', 'x', 'judge'), 'it does not ship', CTX).passed
    assert not score_case(Case('a', 'q', 'x', 'judge'), 'it ships', CTX).passed

import json
import pytest
from my_evals.golden import Case
from my_evals.judge import make_judge, parse
from my_evals.calibrate import agreement
def test_parse_good_and_bad():
    assert parse(json.dumps({'verdict': 'pass', 'reason': 'ok'})) == (True, 'ok')
    assert parse(json.dumps({'verdict': 'fail', 'reason': 'bad'})) == (False, 'bad')
    assert parse('not json')[0] is False
def test_make_judge_passes_schema_and_rubric():
    seen = {}
    def fake_ask(prompt, model, system=None, fmt=None):
        seen.update(prompt=prompt, model=model, system=system, fmt=fmt)
        return json.dumps({'verdict': 'pass', 'reason': 'same facts'})
    judge = make_judge('judge-model', fake_ask)
    assert judge(Case('a', 'Q?', 'ref', 'judge'), 'ans') == (True, 'same facts')
    assert 'Reference answer: ref' in seen['prompt'] and seen['fmt']['required'] == ['verdict', 'reason']
def test_agreement():
    assert agreement([1, 1, 0, 0], [1, 0, 0, 0]) == 0.75
    with pytest.raises(ValueError):
        agreement([], [])

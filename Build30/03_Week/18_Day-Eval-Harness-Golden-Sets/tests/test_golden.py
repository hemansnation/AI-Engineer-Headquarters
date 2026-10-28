import pytest
from my_evals.golden import Case, load_golden, save_golden, add_case, dataset_hash
NAMES = {'exact', 'contains'}
def test_roundtrip_and_hash(tmp_path):
    p = tmp_path / 'g.jsonl'
    save_golden(p, [Case('a', 'q1', 'e1', 'exact'), Case('b', 'q2', 'e2', 'contains', ['t'], 's', True)])
    cases = load_golden(p, NAMES)
    assert [c.id for c in cases] == ['a', 'b'] and cases[1].must_pass
    h1 = dataset_hash(p)
    add_case(p, Case('c', 'q3', 'e3', 'exact'), NAMES)
    assert dataset_hash(p) != h1
def test_duplicate_and_unknown_scorer(tmp_path):
    p = tmp_path / 'g.jsonl'
    save_golden(p, [Case('a', 'q', 'e', 'exact')])
    with pytest.raises(ValueError):
        add_case(p, Case('a', 'q', 'e', 'exact'), NAMES)
    with pytest.raises(ValueError):
        add_case(p, Case('z', 'q', 'e', 'magic'), NAMES)

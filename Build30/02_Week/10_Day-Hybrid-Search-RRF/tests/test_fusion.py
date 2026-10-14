import sys
sys.path.insert(0, '.')
from my_search.fusion import naive_score_sum, reciprocal_rank_fusion
def test_doc_in_both_lists_wins():
    fused = reciprocal_rank_fusion([['a', 'b', 'c'], ['b', 'a', 'd']], k=60, top_n=4)
    assert fused[0][0] in ('a', 'b')
    assert {d for d, _ in fused[:2]} == {'a', 'b'}
def test_doc_in_one_list_still_returned():
    fused = reciprocal_rank_fusion([['a'], ['z']], top_n=5)
    assert {d for d, _ in fused} == {'a', 'z'}
def test_rrf_formula():
    fused = dict(reciprocal_rank_fusion([['a'], ['a']], k=60, top_n=1))
    assert abs(fused['a'] - 2 / 61) < 1e-9
def test_ties_are_stable():
    first = reciprocal_rank_fusion([['a'], ['b']], top_n=2)
    second = reciprocal_rank_fusion([['a'], ['b']], top_n=2)
    assert first == second
def test_weights_shift_the_winner():
    plain = reciprocal_rank_fusion([['a'], ['b']], top_n=2)
    heavy = reciprocal_rank_fusion([['a'], ['b']], top_n=2, weights=[1.0, 3.0])
    assert heavy[0][0] == 'b'
    assert plain[0][0] == 'a'
def test_naive_sum_is_dominated_by_big_scale():
    vector = [('v_best', 0.9), ('k_best', 0.2)]
    keyword = [('k_best', 12.0), ('v_best', 0.0)]
    assert naive_score_sum([vector, keyword], 1)[0][0] == 'k_best'

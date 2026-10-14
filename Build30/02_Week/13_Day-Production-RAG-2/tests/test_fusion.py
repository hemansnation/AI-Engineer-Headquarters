from rag.fusion import rrf


def test_chunk_in_both_lists_wins():
    vector = [(1, 0.9), (2, 0.8), (3, 0.7)]
    keyword = [(3, 12.0), (1, 9.0), (4, 5.0)]
    fused = rrf([vector, keyword])
    assert fused[0][0] == 1


def test_score_formula():
    fused = rrf([[(1, 0.9)], [(1, 5.0)]], k=60)
    assert abs(fused[0][1] - (1 / 61 + 1 / 61)) < 1e-9


def test_single_list_keeps_order():
    fused = rrf([[(5, 0.9), (7, 0.8)]])
    assert [doc_id for doc_id, _ in fused] == [5, 7]

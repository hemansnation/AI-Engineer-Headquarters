from my_evals.textdist import levenshtein, similarity, jaccard, edit_table
def test_classic():
    assert levenshtein('kitten', 'sitting') == 3
def test_empty():
    assert levenshtein('', 'abc') == 3
    assert similarity('', '') == 1.0
def test_symmetry():
    assert levenshtein('flaw', 'lawn') == levenshtein('lawn', 'flaw') == 2
def test_table_corner_matches():
    assert edit_table('kitten', 'sitting')[-1][-1] == 3
def test_jaccard():
    assert jaccard('a b c', 'b c d') == 0.5

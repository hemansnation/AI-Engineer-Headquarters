from budget.knapsack import knapsack_select, greedy_by_density, greedy_by_score, context_budget, stretch
def total(ids, values):
    return sum(values[i] for i in ids)
def test_dp_beats_greedy_on_the_classic_trap():
    weights = [6, 5, 5]
    values = [10, 7, 7]
    assert total(greedy_by_density(weights, values, 10), values) == 10
    best = knapsack_select(weights, values, 10)
    assert best == [1, 2]
    assert total(best, values) == 14
def test_budget_is_never_broken():
    weights = [30, 45, 12, 80, 25, 60]
    values = [0.9, 0.8, 0.4, 0.95, 0.5, 0.7]
    for pick in (knapsack_select, greedy_by_density, greedy_by_score):
        ids = pick(weights, values, 100)
        assert sum(weights[i] for i in ids) <= 100
def test_dp_is_never_worse_than_greedy():
    weights = [30, 45, 12, 80, 25, 60]
    values = [0.9, 0.8, 0.4, 0.95, 0.5, 0.7]
    dp = total(knapsack_select(weights, values, 100), values)
    assert dp >= total(greedy_by_density(weights, values, 100), values)
    assert dp >= total(greedy_by_score(weights, values, 100), values)
def test_edge_cases():
    assert knapsack_select([], [], 100) == []
    assert knapsack_select([50], [1.0], 10) == []
    assert knapsack_select([5], [-0.2], 10) == []
    assert knapsack_select([5], [1.0], 0) == []
def test_context_budget_leaves_room():
    assert context_budget(4096, 300, 100, 500, safety=0.0) == 3196
    assert context_budget(100, 80, 30, 10) == 0
def test_stretch_spreads_close_scores():
    out = stretch([0.61, 0.62, 0.70])
    assert min(out) == 0.05
    assert max(out) == 1.0
    assert stretch([0.5, 0.5]) == [1.0, 1.0]

def context_budget(window, system_tokens, question_tokens, answer_tokens, safety=0.05):
    free = window - system_tokens - question_tokens - answer_tokens
    return max(0, int(free * (1 - safety)))
def _usable(weights, values, budget):
    return [i for i, (w, v) in enumerate(zip(weights, values)) if v > 0 and 0 < w <= budget]
def greedy_by_score(weights, values, budget):
    order = sorted(_usable(weights, values, budget), key=lambda i: values[i], reverse=True)
    chosen = []
    used = 0
    for i in order:
        if used + weights[i] <= budget:
            chosen.append(i)
            used += weights[i]
    return sorted(chosen)
def greedy_by_density(weights, values, budget):
    order = sorted(_usable(weights, values, budget), key=lambda i: values[i] / weights[i], reverse=True)
    chosen = []
    used = 0
    for i in order:
        if used + weights[i] <= budget:
            chosen.append(i)
            used += weights[i]
    return sorted(chosen)
def knapsack_select(weights, values, budget):
    ids = _usable(weights, values, budget)
    n = len(ids)
    table = [[0.0] * (budget + 1) for _ in range(n + 1)]
    for row, i in enumerate(ids, start=1):
        w, v = weights[i], values[i]
        for cap in range(budget + 1):
            table[row][cap] = table[row - 1][cap]
            if w <= cap:
                take = table[row - 1][cap - w] + v
                if take > table[row][cap]:
                    table[row][cap] = take
    chosen = []
    cap = budget
    for row in range(n, 0, -1):
        if table[row][cap] != table[row - 1][cap]:
            i = ids[row - 1]
            chosen.append(i)
            cap -= weights[i]
    return sorted(chosen)
def select_chunks(chunks, scores, budget, method='dp'):
    weights = [c.token_count for c in chunks]
    pick = {'dp': knapsack_select, 'density': greedy_by_density, 'score': greedy_by_score}[method]
    ids = pick(weights, scores, budget)
    picked = [chunks[i] for i in ids]
    return sorted(picked, key=lambda c: (c.source, c.index))
def stretch(scores, floor=0.05):
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [1.0] * len(scores)
    return [floor + (1 - floor) * (s - lo) / (hi - lo) for s in scores]

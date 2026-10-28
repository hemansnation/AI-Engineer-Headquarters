import csv
from dataclasses import dataclass
@dataclass
class Feature:
    name: str
    value: int
    low: int
    likely: int
    high: int
    required: bool
def load_features(path):
    with open(path, newline='', encoding='utf-8') as file:
        return [Feature(row['name'], int(row['value']), int(row['low']), int(row['likely']), int(row['high']), row['required'].strip().lower() == 'yes') for row in csv.DictReader(file)]
def pick_features(features, capacity):
    required = [f for f in features if f.required]
    optional = [f for f in features if not f.required]
    left = capacity - sum(f.likely for f in required)
    if left < 0:
        raise ValueError('Required features alone are over budget. Raise the budget or cut a requirement.')
    n = len(optional)
    best = [[0] * (left + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        item = optional[i - 1]
        for w in range(left + 1):
            best[i][w] = best[i - 1][w]
            if item.likely <= w:
                best[i][w] = max(best[i][w], best[i - 1][w - item.likely] + item.value)
    chosen = []
    w = left
    for i in range(n, 0, -1):
        if best[i][w] != best[i - 1][w]:
            chosen.append(optional[i - 1])
            w -= optional[i - 1].likely
    chosen.reverse()
    later = [f for f in optional if f not in chosen]
    return required + chosen, later

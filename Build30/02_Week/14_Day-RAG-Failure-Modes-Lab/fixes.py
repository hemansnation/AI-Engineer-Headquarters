from retrieval import tokenize
PLANS = ('free', 'pro', 'enterprise')
def detect_plan(question):
    words = set(tokenize(question))
    for plan in PLANS:
        if plan in words:
            return plan
    return None
def allowed_chunks(chunks, plan):
    ok = set()
    for i, c in enumerate(chunks):
        if not c['current']:
            continue
        if plan and c['plan'] not in (plan, 'all'):
            continue
        ok.add(i)
    return ok
def edge_first(items):
    out = [None] * len(items)
    left, right = 0, len(items) - 1
    for j, item in enumerate(items):
        if j % 2 == 0:
            out[left] = item
            left += 1
        else:
            out[right] = item
            right -= 1
    return out
def calibrate(inside, outside):
    pool = sorted(set(inside + outside))
    if len(pool) < 2:
        return pool[0], 0.0
    best_t, best_acc = pool[0], -1.0
    for a, b in zip(pool, pool[1:]):
        t = (a + b) / 2
        hit = sum(s >= t for s in inside) / len(inside)
        reject = sum(s < t for s in outside) / len(outside)
        acc = (hit + reject) / 2
        if acc > best_acc:
            best_t, best_acc = t, acc
    return best_t, best_acc

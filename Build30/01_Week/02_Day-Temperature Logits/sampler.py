import math
import random

def softmax(logits, temperature=1.0):
    if temperature <= 0:
        raise ValueError('temperature must be above 0, use greedy_index for 0')
    scaled = [x / temperature for x in logits]
    biggest = max(scaled)
    exps = [math.exp(x - biggest) for x in scaled]
    total = sum(exps)
    return [e / total for e in exps]

def greedy_index(logits):
    return max(range(len(logits)), key=lambda i: logits[i])

def top_k_filter(probs, k):
    if k is None or k >= len(probs):
        return list(probs)
    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    keep = set(order[:k])
    kept = [probs[i] if i in keep else 0.0 for i in range(len(probs))]
    total = sum(kept)
    return [x / total for x in kept]

def top_p_filter(probs, p):
    if p is None or p >= 1.0:
        return list(probs)
    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    keep = set()
    running = 0.0
    for i in order:
        keep.add(i)
        running += probs[i]
        if running >= p:
            break
    kept = [probs[i] if i in keep else 0.0 for i in range(len(probs))]
    total = sum(kept)
    return [x / total for x in kept]

def pick_token(tokens, logits, temperature=1.0, top_k=None, top_p=None, rng=None):
    rng = rng or random.Random()
    if temperature == 0:
        return tokens[greedy_index(logits)]
    probs = softmax(logits, temperature)
    probs = top_k_filter(probs, top_k)
    probs = top_p_filter(probs, top_p)
    return rng.choices(tokens, weights=probs, k=1)[0]
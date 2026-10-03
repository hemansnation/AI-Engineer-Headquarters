import random
from collections import Counter

from sampler import softmax, top_p_filter, pick_token

TOKENS = ['Paris', 'Lyon', 'beautiful', 'a', 'Berlin']
LOGITS = [6.0, 3.5, 3.0, 2.0, 1.0]

print('Toy logits (made up by me, not from a real model)')
for token, logit in zip(TOKENS, LOGITS):
    print(f'  {token:<10} {logit}')

print()
print('Probabilities at three temperatures')
print('token         T=0.2    T=1.0    T=2.0')
cold = softmax(LOGITS, 0.2)
normal = softmax(LOGITS, 1.0)
hot = softmax(LOGITS, 2.0)
for i, token in enumerate(TOKENS):
    print(f'{token:<10} {cold[i]:8.3f} {normal[i]:8.3f} {hot[i]:8.3f}')

print()
print('Top-p 0.90 applied on the T=1.0 probabilities')
cut = top_p_filter(normal, 0.90)
for i, token in enumerate(TOKENS):
    print(f'{token:<10} {normal[i]:8.3f} -> {cut[i]:8.3f}')

print()
print('1000 draws at each temperature, seed 7')
for temperature in (0, 0.2, 1.0, 2.0):
    rng = random.Random(7)
    draws = [pick_token(TOKENS, LOGITS, temperature, rng=rng) for _ in range(1000)]
    counts = Counter(draws)
    line = '  '.join(f'{t}:{counts.get(t, 0)}' for t in TOKENS)
    print(f'T={temperature:<4} {line}')
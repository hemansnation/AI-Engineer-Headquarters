import sys
sys.path.insert(0, '.')
import evaluate
for k in [1, 10, 60, 200]:
    print(f'k = {k}')
    evaluate.run(k=k)

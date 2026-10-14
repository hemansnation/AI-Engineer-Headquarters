import json
import os
from config import TOP_K, LOG_DIR
from corpus import DOCS
from golden_set import GOLDEN
from pipeline import Pipeline, NAIVE, FIXED
from report import print_table
def write_log(name, rows):
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f'run_log_{name}.jsonl')
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
    return path
def main():
    naive = Pipeline(DOCS, NAIVE)
    fixed = Pipeline(DOCS, FIXED)
    acc = fixed.fit_gate(GOLDEN, TOP_K)
    print(f'gate threshold {fixed.threshold:.3f}, balanced accuracy {acc:.2f}')
    naive_rows = [naive.run(item, TOP_K) for item in GOLDEN]
    fixed_rows = [fixed.run(item, TOP_K) for item in GOLDEN]
    print('saved', write_log('naive', naive_rows))
    print('saved', write_log('fixed', fixed_rows))
    print_table(DOCS, GOLDEN, naive_rows, fixed_rows)
if __name__ == '__main__':
    main()

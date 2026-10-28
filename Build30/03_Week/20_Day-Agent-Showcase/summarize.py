import heapq
import json
import math
import sys
from collections import Counter
def percentile(values, p):
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]
def main(path='logs/runs.jsonl'):
    statuses, latencies, slowest = Counter(), [], []
    prompt_tokens = completion_tokens = 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            run = json.loads(line)
            statuses[run['status']] += 1
            latencies.append(run['latency_s'])
            prompt_tokens += run['prompt_tokens']
            completion_tokens += run['completion_tokens']
            item = (run['latency_s'], run['run_id'], run['task'][:50])
            if len(slowest) < 3:
                heapq.heappush(slowest, item)
            else:
                heapq.heappushpop(slowest, item)
    total = sum(statuses.values())
    if total == 0:
        print('No runs yet.')
        return
    print('runs: ' + str(total))
    for status, n in statuses.most_common():
        print('  ' + status + ': ' + str(n) + ' (' + format(n / total, '.0%') + ')')
    print('p50: ' + str(percentile(latencies, 50)) + ' s  p95: ' + str(percentile(latencies, 95)) + ' s')
    print('tokens: ' + str(prompt_tokens) + ' in, ' + str(completion_tokens) + ' out')
    print('slowest 3:')
    for seconds, run_id, task in sorted(slowest, reverse=True):
        print('  ' + run_id + ' ' + str(seconds) + ' s ' + task)
if __name__ == '__main__':
    main(*sys.argv[1:])

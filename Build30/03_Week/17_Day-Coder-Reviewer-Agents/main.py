import argparse
import json
import os
from agents import Coder, Reviewer
from config import MAX_ROUNDS, STALL_THRESHOLD, TOKEN_BUDGET
from coordinator import Coordinator
def build(use_embeddings):
    embed = None
    if use_embeddings:
        from stall import default_embed
        embed = default_embed
    return Coordinator(Coder(), Reviewer(), MAX_ROUNDS, TOKEN_BUDGET, embed, STALL_THRESHOLD)
def show(result):
    for item in result.history.items:
        print(f'Round {item.round}: {item.verdict} ({item.tokens} tokens)')
        if item.feedback:
            print(item.feedback)
    print(f'Stopped because: {result.stop_reason}')
    print(f'Total tokens: {result.tracker.total} of {result.tracker.budget}')
    print('Final code:')
    print(result.code)
def save(task, result):
    os.makedirs('logs', exist_ok=True)
    row = {'task': task, 'stop_reason': result.stop_reason, 'tokens': result.tracker.total, 'rounds': result.history.as_dicts()}
    with open('logs/run_log.jsonl', 'a') as file:
        file.write(json.dumps(row) + '\n')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task')
    parser.add_argument('--stall', action='store_true', help='use the embedding model to detect a stuck loop')
    args = parser.parse_args()
    outcome = build(args.stall).run(args.task)
    show(outcome)
    save(args.task, outcome)

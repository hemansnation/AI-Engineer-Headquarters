from agents import CODER_SYSTEM, Coder, Reviewer, default_ask
from checks import static_issues
from config import MAX_ROUNDS, TOKEN_BUDGET
from coordinator import Coordinator
from criteria import CRITERIA
from parsing import extract_code
TASKS = [
    'Write is_palindrome(text) that ignores case, spaces and punctuation.',
    'Write merge_sorted(a, b) that merges two sorted lists in O(n + m).',
    'Write parse_price(text) that turns $1,499.50 into the float 1499.5 and returns None for bad input.',
]
def single_shot(task, ask_fn):
    prompt = f'Task:\n{task}\n\nFollow these criteria while you write:\n{CRITERIA}'
    text, usage = ask_fn(CODER_SYSTEM, prompt)
    code = extract_code(text)
    return code, sum(usage), len(static_issues(code))
def compare(tasks, ask_fn=None):
    ask_fn = ask_fn or default_ask()
    pair = Coordinator(Coder(ask_fn), Reviewer(ask_fn), MAX_ROUNDS, TOKEN_BUDGET)
    rows = []
    for task in tasks:
        _, single_tokens, single_issues = single_shot(task, ask_fn)
        result = pair.run(task)
        pair_issues = len(static_issues(result.code))
        rows.append((task[:38], single_tokens, single_issues, result.tracker.total, pair_issues, result.stop_reason))
    return rows
if __name__ == '__main__':
    print('task | single tokens | single issues | pair tokens | pair issues | stop')
    for row in compare(TASKS):
        print(' | '.join(str(cell) for cell in row))

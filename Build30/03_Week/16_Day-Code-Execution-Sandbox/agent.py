import re
from llm import ask
from sandbox import run_python
FENCE = chr(96) * 3
RULES = (
    'You write Python 3 code that solves the task. '
    'Use only the standard library. No network, no files outside the current folder. '
    'Print the final answer with print(). '
    'Reply with one python code block and nothing else.'
)
def extract_code(text: str) -> str:
    match = re.search(FENCE + r'(?:python)?\n(.*?)' + FENCE, text, re.DOTALL)
    return match.group(1) if match else text.strip()
def build_prompt(task: str, history: list) -> str:
    parts = [RULES, 'TASK: ' + task]
    for number, (code, feedback) in enumerate(history, start=1):
        parts.append('ATTEMPT ' + str(number) + ' CODE:\n' + code)
        parts.append('ATTEMPT ' + str(number) + ' RESULT:\n' + feedback)
    if history:
        parts.append('Fix the problem and send the full corrected code.')
    return '\n\n'.join(parts)
def solve(task: str, max_rounds: int = 4):
    history = []
    for round_no in range(1, max_rounds + 1):
        reply = ask(build_prompt(task, history))
        code = extract_code(reply.output_text)
        result = run_python(code)
        print('round', round_no, '|', result.backend, '|', 'ok' if result.ok else 'failed', '|', result.duration_s, 's')
        history.append((code, result.for_model()))
        if result.ok:
            print(result.stdout.strip())
            return result
    print('gave up after', max_rounds, 'rounds')
    return None

import json
from .config import JUDGE_MODEL
from .llm import ask
SCHEMA = {
    'type': 'object',
    'properties': {'verdict': {'type': 'string', 'enum': ['pass', 'fail']}, 'reason': {'type': 'string'}},
    'required': ['verdict', 'reason'],
}
RUBRIC = '''You grade a support answer against a reference answer.
Pass only if the answer states the same facts as the reference.
Fail if the answer adds a fact that contradicts the reference or invents a policy.
If the reference says the information is not available, the answer must also say it does not know.
Ignore tone and length. Judge facts only.'''
def build_prompt(question, expected, output):
    return f'Question: {question}\nReference answer: {expected}\nAnswer to grade: {output}'
def parse(raw):
    try:
        data = json.loads(raw)
        return data['verdict'] == 'pass', str(data.get('reason', ''))
    except (ValueError, KeyError, TypeError):
        return False, 'judge returned invalid json'
def make_judge(model=JUDGE_MODEL, ask_fn=ask):
    def judge(case, output):
        raw = ask_fn(build_prompt(case.question, case.expected, output), model, system=RUBRIC, fmt=SCHEMA)
        return parse(raw)
    return judge

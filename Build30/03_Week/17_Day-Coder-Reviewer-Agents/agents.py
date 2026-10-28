from config import REVIEWER_MODEL
from criteria import CRITERIA
from parsing import FENCE, extract_code, parse_verdict
CODER_SYSTEM = 'You are a careful Python coder. Reply with one python code block and nothing else. No explanation.'
REVIEWER_SYSTEM = (
    'You are a strict code reviewer. You never write code. '
    'Your first line must be exactly VERDICT: APPROVED or VERDICT: REVISE. '
    'After that, list at most 3 issues, one per line, each starting with a dash. '
    'Each issue must say what is wrong and how to fix it. '
    'Approve when every criterion passes. Do not invent criteria that are not in the list.'
)
def default_ask(model=None):
    from llm import ask
    def call(system, prompt, temperature=0.2):
        return ask(system, prompt, temperature, model)
    return call
class Coder:
    def __init__(self, ask_fn=None):
        self.ask = ask_fn or default_ask()
    def draft(self, task, previous_code=None, feedback=None):
        if previous_code is None:
            prompt = f'Task:\n{task}'
        else:
            prompt = (
                f'Task:\n{task}\n\nYour previous code:\n{FENCE}python\n{previous_code}\n{FENCE}\n\n'
                f'Reviewer feedback:\n{feedback}\n\nFix every issue and return the full corrected code.'
            )
        text, usage = self.ask(CODER_SYSTEM, prompt)
        return extract_code(text), usage
class Reviewer:
    def __init__(self, ask_fn=None):
        self.ask = ask_fn or default_ask(REVIEWER_MODEL)
    def review(self, task, code):
        prompt = (
            f'Task:\n{task}\n\nCriteria:\n{CRITERIA}\n\n'
            f'Code to review:\n{FENCE}python\n{code}\n{FENCE}'
        )
        text, usage = self.ask(REVIEWER_SYSTEM, prompt)
        verdict, feedback = parse_verdict(text)
        return verdict, feedback, usage

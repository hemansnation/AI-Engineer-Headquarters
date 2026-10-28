from dataclasses import dataclass
from checks import static_issues
from history import History, Revision
from stall import is_stalled
from tracker import Tracker
@dataclass
class Result:
    code: str
    approved: bool
    stop_reason: str
    history: History
    tracker: Tracker
class Coordinator:
    def __init__(self, coder, reviewer, max_rounds=3, token_budget=6000, embed=None, stall_threshold=0.97):
        self.coder = coder
        self.reviewer = reviewer
        self.max_rounds = max_rounds
        self.token_budget = token_budget
        self.embed = embed
        self.stall_threshold = stall_threshold
    def run(self, task):
        history = History()
        tracker = Tracker(self.token_budget)
        code = None
        feedback = None
        stop_reason = 'cap'
        for round_no in range(1, self.max_rounds + 1):
            previous = history.last()
            code, coder_usage = self.coder.draft(task, code, feedback)
            tracker.add(*coder_usage)
            round_prompt, round_completion = coder_usage
            issues = static_issues(code)
            if issues:
                verdict, feedback = 'REVISE', '\n'.join('- ' + issue for issue in issues)
            else:
                verdict, feedback, reviewer_usage = self.reviewer.review(task, code)
                tracker.add(*reviewer_usage)
                round_prompt += reviewer_usage[0]
                round_completion += reviewer_usage[1]
            history.add(Revision(round_no, code, verdict, feedback, round_prompt, round_completion))
            if verdict == 'APPROVED':
                stop_reason = 'approved'
                break
            if previous and is_stalled(previous.code, code, self.embed, self.stall_threshold):
                stop_reason = 'stalled'
                break
            if not tracker.can_afford(history.last().tokens):
                stop_reason = 'budget'
                break
        return Result(code, stop_reason == 'approved', stop_reason, history, tracker)

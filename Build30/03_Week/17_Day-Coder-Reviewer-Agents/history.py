from dataclasses import dataclass, field
@dataclass
class Revision:
    round: int
    code: str
    verdict: str
    feedback: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    @property
    def tokens(self):
        return self.prompt_tokens + self.completion_tokens
@dataclass
class History:
    items: list = field(default_factory=list)
    def add(self, revision):
        self.items.append(revision)
    def last(self):
        return self.items[-1] if self.items else None
    def total_tokens(self):
        return sum(item.tokens for item in self.items)
    def as_dicts(self):
        return [
            {'round': i.round, 'verdict': i.verdict, 'feedback': i.feedback, 'tokens': i.tokens, 'code': i.code}
            for i in self.items
        ]

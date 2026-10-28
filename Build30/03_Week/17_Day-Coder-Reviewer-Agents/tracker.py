class Tracker:
    def __init__(self, budget):
        self.budget = budget
        self.prompt = 0
        self.completion = 0
        self.calls = 0
    def add(self, prompt, completion):
        self.prompt += prompt
        self.completion += completion
        self.calls += 1
    @property
    def total(self):
        return self.prompt + self.completion
    def can_afford(self, next_cost):
        return self.total + next_cost <= self.budget

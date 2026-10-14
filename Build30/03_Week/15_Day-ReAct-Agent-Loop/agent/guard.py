from collections import deque
class LoopGuard:
    def __init__(self, window=4, max_repeats=2):
        self.seen = {}
        self.recent = deque(maxlen=window)
        self.max_repeats = max_repeats
    def fingerprint(self, tool, tool_input):
        return tool + '|' + ' '.join(tool_input.lower().split())
    def check(self, tool, tool_input):
        key = self.fingerprint(tool, tool_input)
        self.seen[key] = self.seen.get(key, 0) + 1
        self.recent.append(key)
        if self.seen[key] > self.max_repeats:
            times = str(self.seen[key] - 1)
            return 'You already called ' + tool + ' with this input ' + times + ' times. Use a different input or write the Final Answer.'
        if len(self.recent) == self.recent.maxlen and self.is_cycle():
            return 'You are going in circles between the same calls. Stop and write the Final Answer with what you know.'
        return None
    def is_cycle(self):
        items = list(self.recent)
        half = len(items) // 2
        same_halves = items[:half] == items[half:half * 2]
        return half > 0 and same_halves and len(set(items)) < len(items)

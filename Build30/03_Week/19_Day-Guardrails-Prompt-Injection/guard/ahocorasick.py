from collections import deque
class AhoCorasick:
    def __init__(self, patterns):
        self.goto = [{}]
        self.fail = [0]
        self.output = [[]]
        for pattern in patterns:
            self._add(pattern)
        self._build()
    def _add(self, pattern):
        node = 0
        for ch in pattern:
            if ch not in self.goto[node]:
                self.goto.append({})
                self.fail.append(0)
                self.output.append([])
                self.goto[node][ch] = len(self.goto) - 1
            node = self.goto[node][ch]
        self.output[node].append(pattern)
    def _build(self):
        queue = deque(self.goto[0].values())
        while queue:
            node = queue.popleft()
            for ch, child in self.goto[node].items():
                queue.append(child)
                state = self.fail[node]
                while state and ch not in self.goto[state]:
                    state = self.fail[state]
                self.fail[child] = self.goto[state].get(ch, 0)
                self.output[child] = self.output[child] + self.output[self.fail[child]]
    def search(self, text):
        state = 0
        hits = []
        for i, ch in enumerate(text):
            while state and ch not in self.goto[state]:
                state = self.fail[state]
            state = self.goto[state].get(ch, 0)
            for pattern in self.output[state]:
                hits.append((i, pattern))
        return hits

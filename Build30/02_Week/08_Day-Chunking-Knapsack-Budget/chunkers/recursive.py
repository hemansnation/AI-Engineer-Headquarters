from .base import BaseChunker
from .packing import hard_split
SEPARATORS = ['\n\n', '\n', '. ', ' ']
def split_spans(text, start, end, sep):
    spans = []
    pos = start
    while pos < end:
        j = text.find(sep, pos, end)
        if j == -1:
            spans.append((pos, end))
            break
        spans.append((pos, j + len(sep)))
        pos = j + len(sep)
    return spans
class RecursiveChunker(BaseChunker):
    name = 'recursive'
    def split_units(self, text):
        spans = self._pieces(text, 0, len(text), SEPARATORS)
        units = []
        for s, e in spans:
            n = self.counter.count(text[s:e])
            if n > 0:
                units.append((s, e, n))
        return units
    def _pieces(self, text, start, end, seps):
        if self.counter.count(text[start:end]) <= self.size:
            return [(start, end)]
        if not seps:
            return [(s, e) for s, e, _ in hard_split(text, start, end, self.counter, self.size)]
        out = []
        for s, e in split_spans(text, start, end, seps[0]):
            out.extend(self._pieces(text, s, e, seps[1:]))
        return out

import pysbd
from .base import BaseChunker
from .packing import hard_split
_segmenter = pysbd.Segmenter(language='en', clean=False, char_span=True)
class SentenceChunker(BaseChunker):
    name = 'sentence'
    def split_units(self, text):
        units = []
        for span in _segmenter.segment(text):
            s, e = span.start, span.end
            n = self.counter.count(text[s:e])
            if n == 0:
                continue
            if n > self.size:
                units.extend(hard_split(text, s, e, self.counter, self.size))
            else:
                units.append((s, e, n))
        return units

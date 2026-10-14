from .base import BaseChunker
class FixedChunker(BaseChunker):
    name = 'fixed'
    def split_units(self, text):
        return [(s, e, 1) for s, e in self.counter.offsets(text)]

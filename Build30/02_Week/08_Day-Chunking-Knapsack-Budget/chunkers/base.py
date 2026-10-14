from .packing import pack
class BaseChunker:
    name = 'base'
    def __init__(self, counter, size=200, overlap=40):
        if size <= 0:
            raise ValueError('size must be above zero')
        if overlap < 0 or overlap >= size:
            raise ValueError('overlap must be zero or more, and smaller than size')
        self.counter = counter
        self.size = size
        self.overlap = overlap
    def split_units(self, text):
        raise NotImplementedError
    def chunk(self, text, source='doc'):
        units = self.split_units(text)
        return pack(text, units, source, self.name, self.counter, self.size, self.overlap)

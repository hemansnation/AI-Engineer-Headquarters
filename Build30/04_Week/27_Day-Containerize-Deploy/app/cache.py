from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.items = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if key not in self.items:
            self.misses += 1
            return None
        self.items.move_to_end(key)
        self.hits += 1
        return self.items[key]

    def put(self, key, value):
        self.items[key] = value
        self.items.move_to_end(key)
        if len(self.items) > self.capacity:
            self.items.popitem(last=False)

    def stats(self):
        return {'size': len(self.items), 'hits': self.hits, 'misses': self.misses}

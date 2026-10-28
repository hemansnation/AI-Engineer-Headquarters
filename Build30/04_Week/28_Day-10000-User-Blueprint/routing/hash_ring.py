import bisect
import hashlib
def _hash(key):
    return int(hashlib.md5(key.encode()).hexdigest(), 16)
class HashRing:
    def __init__(self, nodes=None, virtual_nodes=100):
        self.virtual_nodes = virtual_nodes
        self.ring = []
        self.owner = {}
        for node in nodes or []:
            self.add_node(node)
    def add_node(self, node):
        for i in range(self.virtual_nodes):
            h = _hash(f'{node}#{i}')
            self.owner[h] = node
            bisect.insort(self.ring, h)
    def remove_node(self, node):
        for i in range(self.virtual_nodes):
            h = _hash(f'{node}#{i}')
            self.owner.pop(h, None)
            idx = bisect.bisect_left(self.ring, h)
            if idx < len(self.ring) and self.ring[idx] == h:
                self.ring.pop(idx)
    def get_node(self, key):
        if not self.ring:
            raise ValueError('ring is empty')
        idx = bisect.bisect(self.ring, _hash(key)) % len(self.ring)
        return self.owner[self.ring[idx]]

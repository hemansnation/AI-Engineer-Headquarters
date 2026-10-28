import time
class TokenBucket:
    def __init__(self, capacity, refill_per_sec, clock=time.monotonic):
        self.capacity = capacity
        self.refill_per_sec = refill_per_sec
        self.tokens = float(capacity)
        self.clock = clock
        self.last = clock()
    def allow(self, cost=1):
        now = self.clock()
        elapsed = now - self.last
        self.last = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_sec)
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False
class PerUserLimiter:
    def __init__(self, capacity, refill_per_sec, clock=time.monotonic):
        self.capacity = capacity
        self.refill_per_sec = refill_per_sec
        self.clock = clock
        self.buckets = {}
    def allow(self, user_id, cost=1):
        if user_id not in self.buckets:
            self.buckets[user_id] = TokenBucket(self.capacity, self.refill_per_sec, self.clock)
        return self.buckets[user_id].allow(cost)

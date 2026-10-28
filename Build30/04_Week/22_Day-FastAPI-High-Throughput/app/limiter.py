import time
from dataclasses import dataclass
@dataclass
class Bucket:
    tokens: float
    updated_at: float
class TokenBucketLimiter:
    def __init__(self, capacity: int, refill_per_second: float):
        self.capacity = capacity
        self.refill_per_second = refill_per_second
        self.buckets: dict[str, Bucket] = {}
    def allow(self, key: str, cost: float = 1.0) -> tuple[bool, float]:
        now = time.monotonic()
        bucket = self.buckets.get(key)
        if bucket is None:
            bucket = Bucket(tokens=float(self.capacity), updated_at=now)
            self.buckets[key] = bucket
        elapsed = now - bucket.updated_at
        bucket.tokens = min(self.capacity, bucket.tokens + elapsed * self.refill_per_second)
        bucket.updated_at = now
        if bucket.tokens >= cost:
            bucket.tokens -= cost
            return True, 0.0
        wait_seconds = (cost - bucket.tokens) / self.refill_per_second
        return False, wait_seconds

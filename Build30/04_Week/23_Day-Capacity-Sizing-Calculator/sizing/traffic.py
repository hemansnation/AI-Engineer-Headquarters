from dataclasses import dataclass
@dataclass(frozen=True)
class Traffic:
    users: int
    queries_per_user_per_day: float
    avg_prompt_tokens: int
    avg_completion_tokens: int
    active_hours: float = 10.0
    peak_factor: float = 3.0
    avg_latency_seconds: float = 8.0
def queries_per_day(t):
    return t.users * t.queries_per_user_per_day
def avg_qps(t):
    return queries_per_day(t) / (t.active_hours * 3600)
def peak_qps(t):
    return avg_qps(t) * t.peak_factor
def concurrent_requests(t):
    return peak_qps(t) * t.avg_latency_seconds
def tokens_per_day(t):
    queries = queries_per_day(t)
    prompt = queries * t.avg_prompt_tokens
    completion = queries * t.avg_completion_tokens
    return {'prompt': prompt, 'completion': completion, 'total': prompt + completion}

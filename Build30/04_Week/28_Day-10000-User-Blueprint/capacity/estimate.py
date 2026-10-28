from dataclasses import dataclass
@dataclass
class CapacityReport:
    total_users: int
    daily_queries: int
    daily_tokens: int
    average_qps: float
    peak_qps: float
    gpus_needed: int
    monthly_cost_usd: float
def estimate_cluster_capacity(
    total_users: int,
    queries_per_day: int,
    avg_tokens: int,
    active_ratio: float = 0.3,
    peak_factor: float = 5.0,
    working_seconds: int = 8 * 3600,
    tokens_per_sec_per_gpu: int = 1500,
    gpu_cost_per_hour: float = 1.2,
    cache_hit_rate: float = 0.0,
) -> CapacityReport:
    active_users = int(total_users * active_ratio)
    daily_queries = active_users * queries_per_day
    uncached = daily_queries * (1 - cache_hit_rate)
    daily_tokens = int(uncached * avg_tokens)
    average_qps = uncached / working_seconds
    peak_qps = average_qps * peak_factor
    peak_tokens_per_sec = peak_qps * avg_tokens
    gpus_needed = max(1, -(-int(peak_tokens_per_sec) // tokens_per_sec_per_gpu))
    monthly_cost = gpus_needed * gpu_cost_per_hour * 24 * 30
    return CapacityReport(
        total_users=total_users,
        daily_queries=int(daily_queries),
        daily_tokens=daily_tokens,
        average_qps=round(average_qps, 2),
        peak_qps=round(peak_qps, 2),
        gpus_needed=gpus_needed,
        monthly_cost_usd=round(monthly_cost, 2),
    )
if __name__ == '__main__':
    for hit in (0.0, 0.3):
        report = estimate_cluster_capacity(10000, 20, 1500, cache_hit_rate=hit)
        print(f'cache hit rate {hit:.0%}')
        print(report)

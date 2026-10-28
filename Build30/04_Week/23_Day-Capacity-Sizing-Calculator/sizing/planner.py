import math
from dataclasses import dataclass
from sizing.memory import weights_gib, kv_cache_gib
from sizing.traffic import peak_qps, concurrent_requests, tokens_per_day
HOURS_PER_MONTH = 730
DAYS_PER_MONTH = 30
@dataclass(frozen=True)
class Gpu:
    name: str
    memory_gib: float
    price_per_hour: float
    decode_tokens_per_sec: float
    prefill_tokens_per_sec: float
@dataclass(frozen=True)
class Plan:
    weights_gib: float
    kv_gib_at_peak: float
    total_memory_gib: float
    peak_concurrent: int
    gpus_for_memory: int
    gpus_for_decode: int
    gpus_for_prefill: int
    gpus_final: int
    bottleneck: str
    monthly_cost: float
    tokens_per_day: float
    cost_per_million_tokens: float
def make_plan(spec, traffic, gpu, precision='fp16', kv_precision='fp16', gpu_utilization=0.9, target_utilization=0.7, overhead_gib=2.0, redundancy=1):
    weights = weights_gib(spec.params_billion, precision)
    concurrent = math.ceil(concurrent_requests(traffic))
    context = traffic.avg_prompt_tokens + traffic.avg_completion_tokens
    kv = kv_cache_gib(spec, context, concurrent, kv_precision)
    total = weights + kv + overhead_gib
    peak = peak_qps(traffic)
    for_memory = math.ceil(total / (gpu.memory_gib * gpu_utilization))
    for_decode = math.ceil(peak * traffic.avg_completion_tokens / (gpu.decode_tokens_per_sec * target_utilization))
    for_prefill = math.ceil(peak * traffic.avg_prompt_tokens / (gpu.prefill_tokens_per_sec * target_utilization))
    needs = {'memory': for_memory, 'decode speed': for_decode, 'prefill speed': for_prefill}
    bottleneck = max(needs, key=needs.get)
    final = needs[bottleneck] + redundancy
    monthly = final * gpu.price_per_hour * HOURS_PER_MONTH
    daily = tokens_per_day(traffic)['total']
    per_million = monthly / (daily * DAYS_PER_MONTH / 1e6)
    return Plan(weights, kv, total, concurrent, for_memory, for_decode, for_prefill, final, bottleneck, monthly, daily, per_million)

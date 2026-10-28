# Formulas

Prices are USD per 1M tokens; divide by 1,000,000.

## 1. Single request
```
cost = ( T_in_uncached*P_in + T_cache_read*P_in*r + T_cache_write*P_in*w
       + (T_out + T_reasoning)*P_out ) / 1e6
```
r = cache read multiplier, w = cache write multiplier. If batched, multiply by (1 - batch_discount).
Reasoning tokens bill at the output rate.

## 2. Agent loop context growth
Fixed prefix S, d_k tokens appended at turn k:
```
input at turn k = S + sum_{j<k} d_j
total input over K turns = K*S + sum_{k=1..K} sum_{j<k} d_j
constant d: K*S + d*K(K-1)/2   (quadratic in turns)
```
With caching, the prefix is billed at the read rate after the first write.

## 3. Cache breakeven
```
cached   = T*P_in*w + n*T*P_in*r
uncached = (n+1)*T*P_in
caching saves money when n > (w - 1) / (1 - r)
```
w=1.25, r=0.10 -> n > 0.28 (one reuse pays). w=2.0, r=0.10 -> n > 1.11 (need 2 reads).
Poisson arrivals at rate lambda/s, TTL t s (refreshes on hit): expected hits before eviction = e^(lambda*t) - 1.

## 4. Self-host vs API breakeven
```
self_host_monthly = gpu_lease + power_cooling + ops_overhead
effective_capacity_tokens = throughput_tok_s * 3600*24*30 * utilization
self_host_cost_per_token  = self_host_monthly / effective_capacity_tokens
breakeven_tokens_per_month = fixed_monthly / (api_cost_per_token - variable_cost_per_token)
```

## 5. Monte Carlo agent run
Per run (NumPy, seeded): sample K turns (capped at max_turns); per turn sample tool output, reasoning,
visible output (lognormal); retries repeat the turn with p_retry (context not pruned); subagents spawn with
their probability, inheriting parent context if `inherits_context`; cost per turn via formula 1 with
context from formula 2. Return mean, p50, p90, p95, p99, max, histogram, cost share by component.

## 6. Unit economics
```
cost_per_user = total_cost(user) over period
gross_margin  = (revenue - inference_cost - other_cogs) / revenue
cost_per_task = total_cost(feature) / successful_tasks(feature)
```

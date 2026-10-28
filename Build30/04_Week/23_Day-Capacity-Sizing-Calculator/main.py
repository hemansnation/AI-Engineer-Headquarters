import argparse
from rich.console import Console
from rich.table import Table
from sizing.presets import PRESETS, load_from_hub
from sizing.traffic import Traffic, queries_per_day, avg_qps, peak_qps, concurrent_requests
from sizing.planner import Gpu, make_plan
def build_parser():
    p = argparse.ArgumentParser(description='Day 23 capacity calculator')
    p.add_argument('--model', default='llama-3.1-8b', choices=sorted(PRESETS))
    p.add_argument('--hf-model', default=None, help='Hugging Face repo id, overrides --model')
    p.add_argument('--precision', default='fp16', choices=['fp16', 'bf16', 'fp8', 'int8', 'int4'])
    p.add_argument('--kv-precision', default='fp16', choices=['fp16', 'fp8'])
    p.add_argument('--users', type=int, default=10000)
    p.add_argument('--queries-per-user', type=float, default=8)
    p.add_argument('--prompt-tokens', type=int, default=1500)
    p.add_argument('--completion-tokens', type=int, default=300)
    p.add_argument('--active-hours', type=float, default=10)
    p.add_argument('--peak-factor', type=float, default=3)
    p.add_argument('--latency', type=float, default=8, help='seconds per request')
    p.add_argument('--gpu-gib', type=float, default=80)
    p.add_argument('--gpu-price', type=float, default=2.5, help='USD per hour, placeholder')
    p.add_argument('--decode-tps', type=float, default=2500, help='output tokens per second per GPU, placeholder')
    p.add_argument('--prefill-tps', type=float, default=20000, help='input tokens per second per GPU, placeholder')
    return p
def main():
    args = build_parser().parse_args()
    spec = load_from_hub(args.hf_model) if args.hf_model else PRESETS[args.model]
    traffic = Traffic(args.users, args.queries_per_user, args.prompt_tokens, args.completion_tokens, args.active_hours, args.peak_factor, args.latency)
    gpu = Gpu('your-gpu', args.gpu_gib, args.gpu_price, args.decode_tps, args.prefill_tps)
    plan = make_plan(spec, traffic, gpu, args.precision, args.kv_precision)
    table = Table(title='Day 23 capacity plan for ' + spec.name + ' in ' + args.precision)
    table.add_column('Step')
    table.add_column('Value', justify='right')
    table.add_row('Queries per day', f'{queries_per_day(traffic):,.0f}')
    table.add_row('Average QPS', f'{avg_qps(traffic):.2f}')
    table.add_row('Peak QPS', f'{peak_qps(traffic):.2f}')
    table.add_row('Requests in flight at peak', f'{plan.peak_concurrent}')
    table.add_row('Tokens per day', f'{plan.tokens_per_day:,.0f}')
    table.add_row('Weights (GiB)', f'{plan.weights_gib:.1f}')
    table.add_row('KV cache at peak (GiB)', f'{plan.kv_gib_at_peak:.1f}')
    table.add_row('Total memory (GiB)', f'{plan.total_memory_gib:.1f}')
    table.add_row('GPUs for memory', f'{plan.gpus_for_memory}')
    table.add_row('GPUs for decode speed', f'{plan.gpus_for_decode}')
    table.add_row('GPUs for prefill speed', f'{plan.gpus_for_prefill}')
    table.add_row('Bottleneck', plan.bottleneck)
    table.add_row('GPUs to buy (with spare)', f'{plan.gpus_final}')
    table.add_row('Monthly cost (USD)', f'{plan.monthly_cost:,.0f}')
    table.add_row('Cost per 1M tokens (USD)', f'{plan.cost_per_million_tokens:.3f}')
    Console().print(table)
if __name__ == '__main__':
    main()

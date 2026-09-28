import json
import time
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

LOG_FILE = Path(__file__).parent / "token_log.jsonl"

PRICES_PER_MILLION = {
    "gemini-3.1-flash-lite": {"input": 0.30, "output": 2.50},
    "gemini-3.8-flash": {"input": 0.75, "output": 3.75},
}

def estimate_cost(model, input_tokens, output_tokens):
    price = PRICES_PER_MILLION.get(model)
    if price is None:
        return 0.0
    input_cost = (input_tokens / 1_000_000) * price["input"]
    output_cost = (output_tokens / 1_000_000) * price["output"]

    return round(input_cost + output_cost, 8)

def append_record(record, path=LOG_FILE):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def track_tokens(func):
    @wraps(func)
    def wrapper(prompt, *args, **kwargs):
        start = time.perf_counter()
        interaction = func(prompt, *args, **kwargs)
        seconds = round(time.perf_counter() - start, 2)

        usage = interaction.usage
        input_tokens = (usage.total_input_tokens or 0) if usage else 0
        output_tokens = (usage.total_output_tokens or 0) if usage else 0
        thought_tokens = (usage.total_thought_tokens or 0) if usage else 0
        model = interaction.model or "unknown"

        record = {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": model,
            "prompt_preview": prompt[:40],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "thought_tokens": thought_tokens,
            "latency_s": seconds,
            "paid_equivalent_usd": estimate_cost(model, input_tokens, output_tokens + thought_tokens),
        }
        append_record(record)
        return interaction

    return wrapper

def running_totals(path=LOG_FILE):
    totals = {"calls": 0, "tokens": 0, "paid_equivalent_usd": 0.0}
    if not Path(path).exists():
        return totals
    with open(path, encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            totals["calls"] += 1
            totals["tokens"] += record["input_tokens"] + record["output_tokens"] + record["thought_tokens"]
            totals["paid_equivalent_usd"] += record["paid_equivalent_usd"]
    totals["paid_equivalent_usd"] = round(totals["paid_equivalent_usd"],8)
    return totals

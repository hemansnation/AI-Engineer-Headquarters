import math
from config import HOURLY_RATE_USD, OVERHEAD_PCT, HOSTING_USD_PER_MONTH, TOKENS_PER_TASK
def three_point(feature):
    expected = (feature.low + 4 * feature.likely + feature.high) / 6
    spread = (feature.high - feature.low) / 6
    return expected, spread
def round_to_50(amount):
    return int(round(amount / 50) * 50)
def price_scope(features):
    expected = sum(three_point(f)[0] for f in features)
    spread = math.sqrt(sum(three_point(f)[1] ** 2 for f in features))
    hours = {'expected': expected, 'p80': expected + 0.84 * spread, 'p95': expected + 1.65 * spread}
    factor = (1 + OVERHEAD_PCT) * HOURLY_RATE_USD
    prices = {name: round_to_50(value * factor) for name, value in hours.items()}
    return hours, prices
def run_cost(volume, share, workdays=22, peak_factor=3):
    tasks = volume * share
    tokens_per_month = int(tasks * TOKENS_PER_TASK)
    average_per_minute = tasks / workdays / 8 / 60
    return {'tasks_per_month': int(tasks), 'tokens_per_month': tokens_per_month, 'peak_requests_per_minute': round(average_per_minute * peak_factor, 2), 'hosting_usd': HOSTING_USD_PER_MONTH}

"""Decimal money helpers: per-MTok conversion, quantization, USD formatting. Never use float."""

from decimal import ROUND_HALF_UP, Decimal

USD = Decimal

TOKENS_PER_MTOK = Decimal(1_000_000)
_INTERNAL = Decimal("0.000001")
_DISPLAY = Decimal("0.01")


def quantize_internal(value: Decimal) -> Decimal:
    """Round to 6 decimal places for internal math."""
    return value.quantize(_INTERNAL, rounding=ROUND_HALF_UP)


def quantize_display(value: Decimal) -> Decimal:
    """Round to 2 decimal places for display."""
    return value.quantize(_DISPLAY, rounding=ROUND_HALF_UP)


def per_mtok(tokens: int, price_per_mtok: Decimal) -> Decimal:
    """Cost in USD of `tokens` at `price_per_mtok` USD per 1M tokens (unrounded)."""
    if tokens < 0:
        raise ValueError("tokens must be non-negative")
    if price_per_mtok < 0:
        raise ValueError("price must be non-negative")
    return Decimal(tokens) * price_per_mtok / TOKENS_PER_MTOK


def format_usd(value: Decimal) -> str:
    """Format as '$1,234.57'; non-zero values below one cent show as '<$0.01'."""
    if value == 0:
        return "$0.00"
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    if magnitude < _DISPLAY / 2:
        return f"{sign}<$0.01"
    return f"{sign}${quantize_display(magnitude):,}"

from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from llmcost.money import format_usd, per_mtok, quantize_display, quantize_internal

prices = st.decimals(min_value=0, max_value=1000, places=4)
tokens = st.integers(min_value=0, max_value=10**9)


def test_per_mtok_golden() -> None:
    # 2_500_000 tokens * $3.00 / 1M = $7.50
    assert per_mtok(2_500_000, Decimal("3.00")) == Decimal("7.5")


def test_per_mtok_rejects_negative() -> None:
    with pytest.raises(ValueError):
        per_mtok(-1, Decimal("1"))
    with pytest.raises(ValueError):
        per_mtok(1, Decimal("-1"))


@given(tokens, tokens, prices)
def test_per_mtok_linear(a: int, b: int, p: Decimal) -> None:
    assert per_mtok(a, p) + per_mtok(b, p) == per_mtok(a + b, p)


@given(tokens, prices)
def test_per_mtok_non_negative(t: int, p: Decimal) -> None:
    assert per_mtok(t, p) >= 0


def test_quantize() -> None:
    assert quantize_internal(Decimal("1.2345675")) == Decimal("1.234568")
    assert quantize_display(Decimal("1.005")) == Decimal("1.01")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (Decimal("1234.567"), "$1,234.57"),
        (Decimal("0"), "$0.00"),
        (Decimal("0.001"), "<$0.01"),
        (Decimal("0.004999"), "<$0.01"),
        (Decimal("0.005"), "$0.01"),
        (Decimal("-12.5"), "-$12.50"),
    ],
)
def test_format_usd(value: Decimal, expected: str) -> None:
    assert format_usd(value) == expected

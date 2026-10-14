import pytest
from safe_math import UnsafeExpressionError, safe_calculate
def test_basic_math():
    assert safe_calculate('2 + 3 * 4') == 14
    assert safe_calculate('(15 / 100) * 2480 + 37 * 12') == 816
    assert safe_calculate('-5 + 2 ** 3') == 3
@pytest.mark.parametrize('attack', [
    '__import__(\'os\').system(\'ls\')',
    'open(\'/etc/passwd\').read()',
    '[x for x in range(10)]',
    '().__class__',
    'abs(-5)',
    'a + 1',
    '\'text\' * 5',
    '1j + 1',
    'True + 1',
])
def test_attacks_are_blocked(attack):
    with pytest.raises(UnsafeExpressionError):
        safe_calculate(attack)
def test_huge_power_is_blocked():
    with pytest.raises(UnsafeExpressionError):
        safe_calculate('9 ** 9 ** 9')
def test_long_input_is_blocked():
    with pytest.raises(UnsafeExpressionError):
        safe_calculate('1+' * 200 + '1')
def test_divide_by_zero_is_a_normal_error():
    with pytest.raises(ZeroDivisionError):
        safe_calculate('1 / 0')

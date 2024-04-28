"""Module for miscellaneous multi-use functions"""

__all__ = [
    'float_to_str', 'round_half_up',
]

import decimal
from typing import List


def float_to_str(f: float) -> str:
    """
    Converts a float to string without scientific notation
    Args:
        f:
            A floating point value

    Returns:
        str
    """
    float_str = str(f)
    if 'e' not in float_str:
        return float_str

    digits, exp_str = float_str.split('e')
    digits = digits.replace('.', '').replace('-', '')
    exp_int = int(exp_str)
    zero_padding = '0' * (abs(exp_int) - 1)
    sign = '-' if f < 0 else ''
    if exp_int > 0:
        return f'{sign}{digits}{zero_padding}0'

    return f'{sign}0.{zero_padding}{digits}'


def round_half_up(value: float, precision) -> float:
    """
    Rounds numbers to the nearest whole, where a value exactly between the two nearest
    wholes is rounded to the higher whole.

    Args:
        value:
            The float value to be rounded
        precision:
            The precision to round the float value to

    """
    return float(
        decimal.Decimal(float_to_str(value)).quantize(
            decimal.Decimal(10) ** -precision,
            rounding=decimal.ROUND_HALF_UP if value >= 0 else decimal.ROUND_HALF_DOWN
        )
    )


def test_sub_list(list_a: List, list_b: List) -> bool:
    """
    Test whether A is a sublist of B.

    Args:
        list_a: (List)
            A list with elements of Any type

        list_b: (List)
            A second list with elements of Any type

    Returns:
        bool
    """
    if len(list_a) > len(list_b):
        return False

    for i in range(0, len(list_b) - len(list_a) + 1):
        if list_b[i:i+len(list_a)] == list_a:
            return True

    return False

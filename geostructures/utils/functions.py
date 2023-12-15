"""Module for miscellaneous multi-use functions"""

__all__ = ['float_to_str', 'round_half_up', 'round_up', 'round_down']

import decimal
from math import floor, ceil

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

def round_down(x: float, nearest: float = 1.0) -> float:
    """
    Rounds numbers down to the nearest specified float, 
    where a 1.0 rounds down to the nearest whole number and .1
    rounds down to the nearest tenth.

    Args:
        value:
            The float value to be rounded
        nearest:
            The nearest float to round the float value down to

    """
    inv = 1 / nearest
    return floor(x * inv) / inv


def round_up(x: float, nearest: float = 1.0) -> float:
    """
    Rounds numbers up to the nearest specified float, 
    where a 1.0 rounds up to the nearest whole number and .1
    rounds up to the nearest tenth.

    Args:
        value:
            The float value to be rounded
        nearest:
            The nearest float to round the float value up to

    """
    inv = 1 / nearest
    return ceil(x * inv) / inv
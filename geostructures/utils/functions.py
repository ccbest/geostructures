"""Module for miscellaneous multi-use functions"""

__all__ = ['float_to_str', 'round_half_up']

import decimal


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
        return f'{sign}{digits}{zero_padding}'

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
            decimal.Decimal(10) ** -precision, rounding=decimal.ROUND_HALF_UP
        )
    )


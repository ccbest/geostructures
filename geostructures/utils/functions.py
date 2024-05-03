"""Module for miscellaneous multi-use functions"""

__all__ = ['round_half_up']


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
    mod = value + 10 ** -(precision + 12)

    return round(mod, precision)


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

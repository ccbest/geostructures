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
    mod= value + 10 ** -(precision +12)
    
    return round(mod,precision) 

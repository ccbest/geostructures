"""
Module for unit conversions
"""
__all__ = ['convert_to_meters', 'convert_to_mps']


def convert_to_meters(distance: float, unit: str):
    """
    Converts distance to meters.

    Args:
        distance (float): The distance value.
        unit (str): The unit of distance (kilometer= 'km', mile = 'mi'
        , feet ='ft',nautical mile = 'nm', yard = 'yd').

    Returns:
        float: The distance in meters.
    """
    unit = unit.lower()
    conversion_factors = {
        'km': 1000,
        'mi': 1609.34,
        'ft': 0.3048,
        'nmi': 1852,
        'yd': 0.9144,
    }

    if unit in conversion_factors:
        return distance * conversion_factors[unit]


def convert_to_mps(speed: float, unit: str):
    """
    Converts speed from different units to meters per second (m/s).

    Args:
        speed (float): Speed value.
        unit (str): Speed unit (kilometer per hour= 'kph', mile per hour= 'mph',
        knot = 'kn').

    Returns:
        float: Speed in meters per second.
    """
    unit = unit.lower()
    conversion_factors = {
        'kph': 1000,
        'mph': 0.44704,
        'kn': 0.5144444
    }

    if unit in conversion_factors:
        return speed * conversion_factors[unit]

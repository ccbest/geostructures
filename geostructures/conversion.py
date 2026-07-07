"""
Module for unit conversions
"""
__all__ = ['convert_to_meters', 'convert_to_mps']


def convert_to_meters(distance: float, unit: str) -> float:
    """
    Converts distance to meters.

    Args:
        distance (float): The distance value.
        unit (str): The unit of distance (meter = 'm', kilometer = 'km', mile = 'mi',
        feet = 'ft', nautical mile = 'nmi', yard = 'yd').

    Returns:
        float: The distance in meters.

    Raises:
        ValueError: If the unit is not recognized.
    """
    unit = unit.lower()
    conversion_factors = {
        'm': 1,
        'km': 1000,
        'mi': 1609.344,
        'ft': 0.3048,
        'nmi': 1852,
        'yd': 0.9144,
    }

    if unit not in conversion_factors:
        raise ValueError(
            f'Unknown distance unit {unit!r}; expected one of '
            f'{sorted(conversion_factors)}'
        )

    return distance * conversion_factors[unit]


def convert_to_mps(speed: float, unit: str) -> float:
    """
    Converts speed from different units to meters per second (m/s).

    Args:
        speed (float): Speed value.
        unit (str): Speed unit (meters per second = 'mps', kilometer per hour = 'kph',
        mile per hour = 'mph', knot = 'kn').

    Returns:
        float: Speed in meters per second.

    Raises:
        ValueError: If the unit is not recognized.
    """
    unit = unit.lower()
    conversion_factors = {
        'mps': 1,
        'kph': 1000 / 3600,
        'mph': 0.44704,
        'kn': 1852 / 3600,
    }

    if unit not in conversion_factors:
        raise ValueError(
            f'Unknown speed unit {unit!r}; expected one of '
            f'{sorted(conversion_factors)}'
        )

    return speed * conversion_factors[unit]

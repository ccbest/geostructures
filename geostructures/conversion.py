"""
Module for unit conversions
"""
__all__ = ['convert_to_meters', 'convert_to_mps']

def convert_to_meters(distance, unit):
    """
    Converts distance to meters.

    Args:
        distance (float): The distance value.
        unit (str): The unit of distance (e.g., 'km', 'mi', 'ft', 'yd').

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
    else:
        raise ValueError(f"Unsupported unit: {unit}")
    

def convert_to_mps(speed, unit):
    """
    Converts speed from different units to meters per second (m/s).

    Args:
        speed (float): Speed value.
        unit (str): Speed unit (e.g., 'kmh', 'mph', 'kn').

    Returns:
        float: Speed in meters per second.
    """
    unit = unit.lower()
    conversion_factors = {
        'kmh': 1000,
        'mph': 0.44704,
        'kn': 0.5144444
    }

    if unit in conversion_factors:
        return speed * conversion_factors[unit]
    else:
        raise ValueError(f"Unsupported unit: {unit}")
    


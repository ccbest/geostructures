import pytest
from  geostructures.conversion import *

def test_convert_to_meters():
    # Test cases: (distance, unit, expected_result)
    test_data = [
        (1.0, 'km', 1000.0),
        (1.0, 'mi', 1609.34),
        (1.0, 'ft', 0.3048),
        (1.0, 'nmi', 1852.0),
        (1.0, 'yd', 0.9144),
    ]

    for distance, unit, expected_result in test_data:
        result = convert_to_meters(distance, unit)
        assert result == pytest.approx(expected_result, rel=1e-6)

def test_convert_to_mps():
    # Test cases: (speed, unit, expected_result)
    test_data = [
        (1.0, 'kmh', 1000.0),
        (1.0, 'mph', 0.44704),
        (1.0, 'kn', 0.5144444),
    ]

    for speed, unit, expected_result in test_data:
        result = convert_to_mps(speed, unit)
        assert result == pytest.approx(expected_result, rel=1e-6)

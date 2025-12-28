

from geostructures.calc import *
from geostructures._geometry import *
from geostructures.coordinates import Coordinate


def test_rotate_coordinates():
    points = [
        Coordinate(1.0, 0.0),
        Coordinate('1.000', '0.000'),
        Coordinate('1.0', '0.000'),
    ]
    result = rotate_coordinates(points, Coordinate(0.0, 0.0), 45)
    assert [Coordinate(round_half_up(x.longitude, 3), round_half_up(x.latitude, 3)) for x in result] == [
        Coordinate(0.707, 0.707),
        Coordinate(0.707, 0.707),
        Coordinate(0.707, 0.707),
    ]

    # Preserve Z values
    points = [
        Coordinate(1.0, 0.0, z=5.),
    ]
    result = rotate_coordinates(points, Coordinate(0.0, 0.0), 45)
    assert result[0].z == 5.

    # Antimeridian test
    points = [
        Coordinate(-179, 0.),
        Coordinate(179, 0.)
    ]
    result = rotate_coordinates(points, Coordinate(179.999, 0.), 135)
    assert [Coordinate(round_half_up(x.longitude, 7), round_half_up(x.latitude, 7)) for x in result] == [
        Coordinate(179.2911861, 0.7078139),
        Coordinate(-179.2946003, -0.7063997)
    ]



from geostructures.calc import *
from geostructures._geometry import *
from geostructures.coordinates import Coordinate


def test_bearing_degrees():
    assert bearing_degrees(Coordinate(0.0, 0.0), Coordinate(0.001, 0.001)) == 45.
    assert bearing_degrees(Coordinate(0.0, 0.0), Coordinate(0.001, 0.001), precision=9) == 44.999999996


def test_haversine_distance_meters():
    # Sourced from haversine package
    actual_dist_meters = 157.25359
    calc_dist = haversine_distance_meters(Coordinate(0.0, 0.0), Coordinate(0.001, 0.001))
    assert round(actual_dist_meters) == round(calc_dist)

    actual_dist_meters = 157_249.59847
    calc_dist = haversine_distance_meters(Coordinate(0.0, 0.0), Coordinate(1.0, 1.0))
    assert abs(round(actual_dist_meters) - round(calc_dist)) < 2

    # Antimeridian test
    actual_dist_meters = 222390
    calc_dist = haversine_distance_meters(Coordinate(179., 0.), Coordinate(-179., 0.))
    assert round(calc_dist) == actual_dist_meters


def test_inverse_haversine_radians():
    assert inverse_haversine_radians(Coordinate(0.0, 0.0), math.radians(45), 111_000) == Coordinate(0.7059029, 0.7058494)


def test_inverse_haversine_degrees():
    assert inverse_haversine_degrees(Coordinate(0.0, 0.0), 45., 111_000) == Coordinate(0.7059029, 0.7058494)


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

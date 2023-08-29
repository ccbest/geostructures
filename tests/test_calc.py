
import math

from geostructures.calc import *
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
    assert rotate_coordinates(points, Coordinate(0.0, 0.0), 45) == [
        Coordinate(0.7, 0.7),
        Coordinate(0.707, 0.707),
        Coordinate(0.707, 0.707),
    ]


def test_find_line_intersection():
    # X-shape, intersect in middle
    line1 = (Coordinate(0.0, 1.0), Coordinate(1.0, 0.0))
    line2 = (Coordinate(0.0, 0.0), Coordinate(1.0, 1.0))
    assert find_line_intersection(line1, line2) == Coordinate(0.5, 0.5)

    # Coordinates run right to left, should get flipped
    line1 = (Coordinate(1.0, 0.0), Coordinate(0.0, 1.0), )
    line2 = (Coordinate(1.0, 1.0), Coordinate(0.0, 0.0), )
    assert find_line_intersection(line1, line2) == Coordinate(0.5, 0.5)

    # Intersect at ends
    line1 = (Coordinate(0.0, 1.0), Coordinate(1.0, 0.0))
    line2 = (Coordinate(1.0, 0.0), Coordinate(2.0, 0.0))
    assert find_line_intersection(line1, line2) == Coordinate(1.0, 0.0)

    # Parallel lines
    line1 = (Coordinate(0.0, 1.0), Coordinate(1.0, 0.0))
    line2 = (Coordinate(1.0, 2.0), Coordinate(2.0, 1.0))
    assert not find_line_intersection(line1, line2)

    # Out of bounds
    line1 = (Coordinate(5.0, 4.0), Coordinate(5.0, 4.0))
    line2 = (Coordinate(1.0, 2.0), Coordinate(2.0, 1.0))
    assert not find_line_intersection(line1, line2)

    # In bounds and not parallel but no intersection
    line1 = (Coordinate(0.0, 0.0), Coordinate(0.5, 0.5))
    line2 = (Coordinate(0.0, 0.5), Coordinate(0.1, 0.3))
    assert not find_line_intersection(line1, line2)



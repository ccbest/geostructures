
from geostructures import Coordinate
from geostructures.distance import *

from tests.functions import assert_coordinates_equal


def test_haversine_distance():
    # Sourced from haversine package
    actual_dist_meters = 157.25359
    calc_dist = haversine_distance(Coordinate(0.0, 0.0), Coordinate(0.001, 0.001))
    assert round(actual_dist_meters) == round(calc_dist)

    actual_dist_meters = 157_249.59847
    calc_dist = haversine_distance(Coordinate(0.0, 0.0), Coordinate(1.0, 1.0))
    assert abs(round(actual_dist_meters) - round(calc_dist)) < 2

    # Antimeridian test
    actual_dist_meters = 222390
    calc_dist = haversine_distance(Coordinate(179., 0.), Coordinate(-179., 0.))
    assert round(calc_dist) == actual_dist_meters


def test_haversine_destination():
    dest = haversine_destination(Coordinate(0.0, 0.0), 45., 111_000)
    assert_coordinates_equal(
        dest,
        Coordinate(0.7059029, 0.7058494),
        test_precision=6,
    )


def test_haversine_bearing():
    bearing = bearing_degrees(Coordinate(0.0, 0.0), Coordinate(0.001, 0.001))
    assert round(bearing, 1) == 45.

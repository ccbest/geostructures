
from geostructures.geodesic import *
from geostructures.coordinates import Coordinate

import pytest
from pytest import approx

from tests.functions import assert_coordinates_equal


def test_haversine_bearing():
    expected = 45.
    actual = haversine_bearing(Coordinate(0.0, 0.0), Coordinate(0.001, 0.001))
    assert actual == approx(expected, abs=1e-6)


def test_haversine_distance():
    # Sourced from haversine package
    expected = 157.253373
    actual = haversine_distance(Coordinate(0.0, 0.0), Coordinate(0.001, 0.001))
    assert expected == approx(actual, abs=1e-6)

    expected = 157_249.381271
    actual = haversine_distance(Coordinate(0.0, 0.0), Coordinate(1.0, 1.0))
    assert expected == approx(actual, abs=1e-6)

    # Antimeridian test
    expected = 222389.853289
    actual = haversine_distance(Coordinate(179., 0.), Coordinate(-179., 0.))
    assert expected == approx(actual, abs=1e-6)


def test_haversine_destination():
    expected = Coordinate(0.7059029, 0.7058494)
    actual = haversine_destination(Coordinate(0.0, 0.0), 45., 111_000)
    assert_coordinates_equal(expected, actual)


def test_vincenty_bearing():
    c1 = Coordinate(0.0, 0.0)
    expected = 45.192423
    actual = vincenty_bearing(c1, Coordinate(0.001, 0.001))
    assert actual == approx(expected, abs=1e-6)

    assert vincenty_bearing(c1, c1) == 0.

    # Follow equator exactly - will trip zero division error
    c1, c2 = Coordinate(0.0, 0.0), Coordinate(1.0, 0.0)
    assert vincenty_bearing(c1, c2) == 90

    # Antipodal test - fall back to haversine
    c1, c2 = Coordinate(0.0, 0.0), Coordinate(180.0, 0.0)
    assert vincenty_bearing(c1, c2) == haversine_bearing(c1, c2)


def test_vincenty_distance():
    # Checked against PyGeodesy library results
    expected = 156.903468
    actual = vincenty_distance(Coordinate(0.0, 0.0), Coordinate(0.001, 0.001))
    assert expected == approx(actual, abs=1e-6)

    expected = 156_899.568291
    actual = vincenty_distance(Coordinate(0.0, 0.0), Coordinate(1.0, 1.0))
    assert expected == approx(actual, abs=1e-6)

    # Antimeridian test
    expected = 222_638.981586
    actual = vincenty_distance(Coordinate(179., 0.), Coordinate(-179., 0.))
    assert expected == approx(actual, abs=1e-6)

    # Same coordinate
    expected = 0.0
    actual = vincenty_distance(Coordinate(0.0, 0.0), Coordinate(0.0, 0.0))
    assert expected == actual

    # Follow equator exactly - will trip ZeroDivisionError
    expected = 111_319.490793
    actual = vincenty_distance(Coordinate(0.0, 0.0), Coordinate(1.0, 0.0))
    assert expected == approx(actual, abs=1e-6)

    # Failure to converge - antipodal
    c1, c2 = Coordinate(0., 0.), Coordinate(180, 0)
    expected = haversine_distance(c1, c2)
    actual = vincenty_distance(c1, c2)
    assert expected == actual


def test_vincenty_destination():
    expected = Coordinate(0.705113, 0.709811)
    actual = vincenty_destination(Coordinate(0.0, 0.0), 45., 111_000)
    assert_coordinates_equal(expected, actual, abs_tol=1e-6)

    c1 = Coordinate(0., 0.)
    assert vincenty_destination(c1, 90., 0.) == c1


def test_karney_bearing():
    expected = 45.192423
    actual = karney_bearing(Coordinate(0.0, 0.0), Coordinate(0.001, 0.001))
    assert actual == approx(expected, abs=1e-6)


def test_karney_distance():
    expected = 156.903471
    actual = karney_distance(Coordinate(0.0, 0.0), Coordinate(0.001, 0.001))
    assert expected == approx(actual, abs=1e-6)

    expected = 156_899.568291
    actual = karney_distance(Coordinate(0.0, 0.0), Coordinate(1.0, 1.0))
    assert expected == approx(actual, abs=1e-6)

    # Antimeridian test
    expected = 222_638.981586
    actual = karney_distance(Coordinate(179., 0.), Coordinate(-179., 0.))
    assert expected == approx(actual, abs=1e-6)

    # Failure to converge - antipodal
    c1, c2 = Coordinate(0., 0.), Coordinate(180, 0)
    expected = haversine_distance(c1, c2)
    actual = vincenty_distance(c1, c2)
    assert expected == actual


def test_karney_destination():
    expected = Coordinate(0.705113, 0.709811)
    actual = karney_destination(Coordinate(0.0, 0.0), 45., 111_000)
    assert_coordinates_equal(expected, actual, abs_tol=1e-6)


def test_set_geodesic_algorithm():
    c1, c2 = Coordinate(0., 0.), Coordinate(0.1, 0.1)
    assert distance_meters(c1, c2) == haversine_distance(c1, c2)
    assert bearing_degrees(c1, c2) == haversine_bearing(c1, c2)
    assert destination_point(c1, 90, 100) == haversine_destination(c1, 90, 100)

    set_geodesic_algorithm('vincenty')
    assert distance_meters(c1, c2) == vincenty_distance(c1, c2)
    assert bearing_degrees(c1, c2) == vincenty_bearing(c1, c2)
    assert destination_point(c1, 90, 100) == vincenty_destination(c1, 90, 100)

    set_geodesic_algorithm('karney')
    assert distance_meters(c1, c2) == karney_distance(c1, c2)
    assert bearing_degrees(c1, c2) == karney_bearing(c1, c2)
    assert destination_point(c1, 90, 100) == karney_destination(c1, 90, 100)

    with pytest.raises(ValueError):
        set_geodesic_algorithm('made up')

    # Clean up - reset to default
    set_geodesic_algorithm('haversine')
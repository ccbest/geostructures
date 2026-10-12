
from datetime import datetime

import pytest
from pytest import approx

from geostructures import GeoBox, GeoCircle, GeoEllipse, GeoPolygon
from geostructures.coordinates import Coordinate
from geostructures.geohash import H3Hasher, NiemeyerHasher

from tests.functions import (
    assert_coordinates_equal, default_test_datetime,
    geojson_round_trip, shapely_round_trip, wkt_round_trip,
)


def test_geocircle_eq():
    circle = GeoCircle(Coordinate(0.0, 0.0), 1000, dt=default_test_datetime)

    c2 = GeoCircle(Coordinate(0.0, 0.0), 1000, dt=default_test_datetime)
    assert c2 == circle

    c2 = GeoCircle(Coordinate(1.0, 1.0), 1000, dt=default_test_datetime)
    assert c2 != circle

    c2 = GeoCircle(Coordinate(0.0, 0.0), 2000, dt=default_test_datetime)
    assert c2 != circle

    c2 = GeoCircle(Coordinate(0.0, 0.0), 1000, dt=datetime(1970, 1, 1, 1, 1))
    assert c2 != circle

    assert 'test' != circle


def test_geocircle_hash():
    circle = GeoCircle(Coordinate(0.0, 0.0), 1000, dt=default_test_datetime)
    c2 = GeoCircle(Coordinate(0.0, 0.0), 1000, dt=default_test_datetime)
    assert len({circle, c2}) == 1


def test_geocircle_repr():
    circle = GeoCircle(Coordinate(0.0, 0.0), 1000, dt=default_test_datetime)
    assert repr(circle) == '<GeoCircle at (0.0, 0.0); radius 1000.0 meters>'


def test_geocircle_bounds():
    actual = GeoCircle(Coordinate(0.0, 0.0), 1000).bounds
    expected = (-0.0089932, -0.0089932, 0.0089932, 0.0089932)
    for b1, b2 in zip(actual, expected):
        assert b1 == approx(b2, abs=1e-6)


def test_geocircle_bounding_coords():
    circle = GeoCircle(Coordinate(0.0, 0.0), 1000)
    expected = [
        Coordinate(-0.0, 0.0089932),
        Coordinate(-0.0015617, 0.0088566),
        Coordinate(-0.0030759, 0.0084509),
        Coordinate(-0.0044966, 0.0077884),
        Coordinate(-0.0057807, 0.0068892)
    ]

    # Verify the first 5 coordinates
    for actual_coord, expected_coord in zip(circle.bounding_coords()[:5], expected):
        assert_coordinates_equal(actual_coord, expected_coord, abs_tol=1e-6)


def test_geocircle_centroid():
    circle = GeoCircle(Coordinate(0.0, 0.0), 1000)
    assert circle.centroid == circle.center


def test_geocircle_contains_coordinate():
    circle = GeoCircle(Coordinate(0.0, 0.0), 1000)

    assert circle.contains_coordinate(Coordinate(0.0, 0.0))
    assert circle.contains_coordinate(Coordinate(0.001, 0.001))
    assert not circle.contains_coordinate(Coordinate(1.0, 1.0))

    circle = GeoCircle(Coordinate(0.0, 0.0), 1000, holes=[GeoCircle(Coordinate(0.0, 0.0), 1000)])
    assert not circle.contains_coordinate(Coordinate(0., 0.))


def test_geocircle_copy():
    circle = GeoCircle(Coordinate(0., 1.), 500)
    circle_copy = circle.copy()

    # Assert equality but different pointer
    assert circle == circle_copy
    assert circle is not circle_copy


def test_geocircle_serialization_round_trips():
    # Circles parse back as their polygon form
    circle = GeoCircle(Coordinate(0.0, 0.0), 1000, dt=default_test_datetime)
    wkt_round_trip(circle)
    geojson_round_trip(circle)
    shapely_round_trip(circle)


def test_geocircle_linear_rings():
    circle = GeoCircle(Coordinate(0.0, 0.0), 1000)

    rings = circle.linear_rings()
    for actual_coord, expected_coord in zip(rings[0], circle.bounding_coords()):
        assert_coordinates_equal(actual_coord, expected_coord, abs_tol=1e-6)

    # Assert self-closing
    assert_coordinates_equal(rings[0][0], rings[0][-1])


def test_geocircle_to_polygon():
    circle = GeoCircle(Coordinate(0.0, 0.0), 1000, dt=default_test_datetime)
    assert circle.to_polygon() == GeoPolygon(circle.bounding_coords(), dt=default_test_datetime)


def test_geocircle_circumscribing_rectangle():
    circle = GeoCircle(Coordinate(0.0, 0.0), 1000, dt=default_test_datetime)
    actual = circle.circumscribing_rectangle()
    expected = GeoBox(
        Coordinate(-0.0089932, 0.0089932),
        Coordinate(0.0089932, -0.0089932),
        dt=default_test_datetime
    )
    assert_coordinates_equal(actual.nw_bound, expected.nw_bound)
    assert_coordinates_equal(actual.se_bound, expected.se_bound)


def test_geocircle_circumscribing_circle():
    circle = GeoCircle(Coordinate(0.0, 0.0), 1000, dt=default_test_datetime)
    assert circle.circumscribing_circle() == circle


def test_geocircle_gaussian_hash_probabilities():
    circle = GeoCircle(Coordinate(-77.03, 38.90), 2000)
    ellipse = GeoEllipse(Coordinate(-77.03, 38.90), 2000, 2000, 45)
    for hasher in (NiemeyerHasher(12, 32), H3Hasher(15)):
        actual = circle.gaussian_hash_probabilities(hasher, 0.95, 0.9, 50)
        expected = ellipse.gaussian_hash_probabilities(hasher, 0.95, 0.9, 50)
        assert actual.keys() == expected.keys()
        assert list(actual.values()) == pytest.approx(list(expected.values()))

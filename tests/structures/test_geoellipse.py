
from datetime import datetime
import math

import numpy as np
from numpy.testing import assert_allclose
import pytest
from pytest import approx

from geostructures import GeoBox, GeoCircle, GeoEllipse, GeoPolygon
from geostructures._geometry import gaussian_probability_in_polygons
from geostructures.coordinates import Coordinate
from geostructures.geohash import H3Hasher, NiemeyerHasher

from tests.functions import (
    assert_coordinates_equal, default_test_datetime,
    geojson_round_trip, shapely_round_trip, wkt_round_trip,
)


def test_geoellipse_eq():
    ellipse = GeoEllipse(Coordinate(0.0, 0.0), 1000, 500, 90, dt=default_test_datetime)

    e2 = GeoEllipse(Coordinate(0.0, 0.0), 1000, 500, 90, dt=default_test_datetime)
    assert e2 == ellipse

    e2 = GeoEllipse(Coordinate(1.0, 0.0), 1000, 500, 90, dt=default_test_datetime)
    assert e2 != ellipse

    e2 = GeoEllipse(Coordinate(0.0, 0.0), 2000, 500, 90, dt=default_test_datetime)
    assert e2 != ellipse

    e2 = GeoEllipse(Coordinate(0.0, 0.0), 1000, 600, 90, dt=default_test_datetime)
    assert e2 != ellipse

    e2 = GeoEllipse(Coordinate(0.0, 0.0), 1000, 500, 180, dt=default_test_datetime)
    assert e2 != ellipse

    e2 = GeoEllipse(Coordinate(0.0, 0.0), 1000, 500, 90, dt=datetime(1970, 1, 1, 1, 1))
    assert e2 != ellipse

    assert 'test' != ellipse


def test_geoellipse_hash():
    ellipse = GeoEllipse(Coordinate(0.0, 0.0), 1000, 500, 90, dt=default_test_datetime)
    e2 = GeoEllipse(Coordinate(0.0, 0.0), 1000, 500, 90, dt=default_test_datetime)
    assert len({e2, ellipse}) == 1


def test_geoellipse_repr():
    ellipse = GeoEllipse(Coordinate(0.0, 0.0), 1000, 500, 90, dt=default_test_datetime)
    assert repr(ellipse) == '<GeoEllipse at (0.0, 0.0); radius 1000.0/500.0; rotation 90.0>'


def test_geoellipse_bounds():
    actual = GeoEllipse(Coordinate(0.0, 0.0), 1000, 500, 45).bounds
    expected = (-0.0071098, -0.0071098, 0.0071098, 0.0071098)
    for b1, b2 in zip(actual, expected):
        approx(b1, b2, abs=1e-6)


def test_geoellipse_bounding_coords():
    ellipse = GeoEllipse(Coordinate(0.0, 0.0), 1000, 500, 90)
    expected = [
        Coordinate(0.0089932, 0.0),
        Coordinate(0.0088586, 0.000775),
        Coordinate(0.0084813, 0.0014955),
        Coordinate(0.0079267, 0.002124),
        Coordinate(0.0072708, 0.0026464)
    ]
    for actual_coord, expected_coord in zip(ellipse.bounding_coords()[:5], expected):
        assert_coordinates_equal(actual_coord, expected_coord, abs_tol=1e-6)

    # assert self-closing
    assert_coordinates_equal(ellipse.bounding_coords()[0], ellipse.bounding_coords()[-1])


def test_geoellipse_centroid():
    ellipse = GeoEllipse(Coordinate(0.0, 0.0), 1000, 500, 90)
    assert ellipse.centroid == ellipse.center


def test_geoellipse_contains():
    ellipse = GeoEllipse(Coordinate(0.0, 0.0), 1000, 500, 90)

    # Center
    assert ellipse.contains_coordinate(Coordinate(0.0, 0.0))

    # 900 meters east
    assert ellipse.contains_coordinate(Coordinate(0.0080939, 0.))

    # 900 meters north (outside)
    assert not ellipse.contains_coordinate(Coordinate(0.0, 0.0080939))

    # 1000 meters east - on edge
    assert ellipse.contains_coordinate(Coordinate(0.0089932, 0.))

    # 45-degree line
    ellipse = GeoEllipse(Coordinate(0.0, 0.0), 1000, 1, 45)
    assert ellipse.contains_coordinate(Coordinate(0.005, 0.005))
    assert ellipse.contains_coordinate(Coordinate(-0.005, -0.005))
    assert not ellipse.contains_coordinate(Coordinate(0, 0.005))
    assert not ellipse.contains_coordinate(Coordinate(-0.005, 0.005))

    # Hole
    ellipse = GeoEllipse(Coordinate(0.0, 0.0), 1000, 500, 90, holes=[GeoCircle(Coordinate(0., 0.), 200)])
    assert not ellipse.contains_coordinate(Coordinate(0., 0))


def test_geoellipse_copy():
    ellipse = GeoEllipse(Coordinate(0., 1.), 500, 200, 90)
    ellipse_copy = ellipse.copy()

    # Assert equality but different pointer
    assert ellipse == ellipse_copy
    assert ellipse is not ellipse_copy


def test_geoellipse_serialization_round_trips():
    # Ellipses parse back as their polygon form
    ellipse = GeoEllipse(Coordinate(0.0, 0.0), 1000, 500, 90, dt=default_test_datetime)
    wkt_round_trip(ellipse)
    geojson_round_trip(ellipse)
    shapely_round_trip(ellipse)


def test_geoellipse_linear_rings():
    ellipse = GeoEllipse(Coordinate(0.0, 0.0), 1000, 500, 90)

    rings = ellipse.linear_rings()
    for actual_coord, expected_coord in zip(rings[0], ellipse.bounding_coords()):
        assert_coordinates_equal(actual_coord, expected_coord, abs_tol=1e-6)

    # Assert self-closing
    assert_coordinates_equal(rings[0][0], rings[0][-1])


def test_geoellipse_to_polygon():
    ellipse = GeoEllipse(Coordinate(0.0, 0.0), 1000, 500, 90, dt=default_test_datetime)
    assert ellipse.to_polygon() == GeoPolygon(ellipse.bounding_coords(), dt=default_test_datetime)


def test_geoellipse_circumscribing_rectangle():
    ellipse = GeoEllipse(Coordinate(0.0, 0.0), 1000, 500, 90, dt=default_test_datetime)

    actual = ellipse.circumscribing_rectangle()
    expected = GeoBox(
        Coordinate(-0.0089932, 0.0044966),
        Coordinate(0.0089932, -0.0044966),
        dt=default_test_datetime
    )
    assert_coordinates_equal(actual.nw_bound, expected.nw_bound)
    assert_coordinates_equal(actual.se_bound, expected.se_bound)


def test_geoellipse_circumscribing_circle():
    ellipse = GeoEllipse(Coordinate(0.0, 0.0), 1000, 500, 90, dt=default_test_datetime)

    assert ellipse.circumscribing_circle() == GeoCircle(
        ellipse.centroid,
        ellipse.semi_major,
        dt=default_test_datetime
    )


def test_geoellipse_covariance_matrix():
    ellipse = GeoEllipse(Coordinate(0., 1.), 100, 50, 45)
    assert_allclose(ellipse.covariance_matrix(), np.array([[6250., 3750.], [3750., 6250.]]))

    ellipse = GeoEllipse(Coordinate(1., 2.), 100, 50, 90)
    assert_allclose(ellipse.covariance_matrix(), np.array([[10000., 0.], [0., 2500.]]))

    ellipse = GeoEllipse(Coordinate(1., 2.), 100, 50, 90)
    assert_allclose(
        ellipse.covariance_matrix(to_trigonometric_rotation=False),
        np.array([[2500., 0.], [0., 10000.]]),
        atol=1e-07
    )


def test_geoellipse_from_covariance_matrix():
    mean = Coordinate(1., 2.)

    cov = np.array([[6250., 3750.], [3750., 6250.]])
    expected = GeoEllipse(Coordinate(1., 2.), 100, 50, 45)
    actual = GeoEllipse.from_covariance_matrix(cov, mean)
    assert expected.centroid == actual.centroid
    for attr in ('semi_major', 'semi_minor', 'rotation'):
        assert_allclose(getattr(expected, attr), getattr(actual, attr))

    cov = np.array([[10000., 0.], [0., 2500.]])
    expected = GeoEllipse(Coordinate(1., 2.), 100, 50, 90)
    actual = GeoEllipse.from_covariance_matrix(cov, mean)
    assert expected.centroid == actual.centroid
    for attr in ('semi_major', 'semi_minor', 'rotation'):
        assert_allclose(getattr(expected, attr), getattr(actual, attr))

    cov = np.array([[2500., 0.], [0., 10000.]])
    expected = GeoEllipse(Coordinate(1., 2.), 100, 50, 90)
    actual = GeoEllipse.from_covariance_matrix(cov, mean, from_trigonometric_rotation=False)
    assert expected.centroid == actual.centroid
    for attr in ('semi_major', 'semi_minor', 'rotation'):
        assert_allclose(getattr(expected, attr), getattr(actual, attr))


def test_geoellipse_gaussian_hash_probabilities():
    import h3

    ellipse = GeoEllipse(Coordinate(-77.03, 38.90), 2000, 800, 30)

    # Achievable at the hasher's resolution
    hasher = NiemeyerHasher(6, 32)
    expected = ellipse.gaussian_hash_probabilities(hasher, 0.95, 0.9, 50)
    assert 1 < len(expected) <= 50
    assert all(len(x) == 6 for x in expected)
    assert sum(expected.values()) >= 0.9
    # The least probable hash was required to reach the mass
    assert sum(expected.values()) - min(expected.values()) < 0.9

    # Hashes are selected most probable first
    k = math.sqrt(-2 * math.log(1 - 0.95))
    neighbors = [y for x in expected for y in hasher.get_neighbors(x) if y not in expected]
    neighbor_probs = gaussian_probability_in_polygons(
        [hasher.get_cell_boundary(x) for x in neighbors],
        ellipse.center, 2000 / k, 800 / k, 30
    )
    assert min(expected.values()) >= max(neighbor_probs)

    # Requires coarsening
    assert ellipse.gaussian_hash_probabilities(NiemeyerHasher(12, 32), 0.95, 0.9, 50) == expected

    actual = ellipse.gaussian_hash_probabilities(H3Hasher(15), 0.95, 0.9, 50)
    assert len(actual) <= 50
    assert {h3.get_resolution(x) for x in actual} == {9}
    assert sum(actual.values()) >= 0.9

    actual = ellipse.gaussian_hash_probabilities(H3Hasher(15), 0.95, 0.9, 1)
    assert len(actual) == 1
    assert sum(actual.values()) >= 0.9

    # A larger contour means a tighter distribution, so fewer hashes are required
    actual = ellipse.gaussian_hash_probabilities(hasher, 0.99, 0.9, 50)
    assert len(actual) < len(expected)

    # Centered on the corner of the four coarsest geohashes, so coarsens past them
    assert GeoEllipse(Coordinate(0., 0.), 2000, 800, 30).gaussian_hash_probabilities(
        NiemeyerHasher(3, 32), 0.95, 0.9, 1
    ) == {'': 1.}
    # Centered on a corner of resolution 1 cells, but within a base cell
    lat, lon = h3.vertex_to_latlng(h3.cell_to_vertexes(h3.latlng_to_cell(0., 0., 1))[0])
    actual = GeoEllipse(Coordinate(lon, lat), 2000, 800, 30).gaussian_hash_probabilities(
        H3Hasher(3), 0.95, 0.9, 1
    )
    assert list(actual) == ['8083fffffffffff']
    assert actual['8083fffffffffff'] == pytest.approx(1.)

    # Centered on a corner of base cells
    lat, lon = h3.vertex_to_latlng(h3.cell_to_vertexes(h3.latlng_to_cell(0., 0., 0))[0])
    assert GeoEllipse(Coordinate(lon, lat), 2000, 800, 30).gaussian_hash_probabilities(
        H3Hasher(3), 0.95, 0.9, 1
    ) == {'': 1.}

    for contour, mass, max_hashes in ((0., 0.9, 1), (1., 0.9, 1), (0.95, 0., 1), (0.95, 1., 1), (0.95, 0.9, 0)):
        with pytest.raises(ValueError):
            ellipse.gaussian_hash_probabilities(hasher, contour, mass, max_hashes)

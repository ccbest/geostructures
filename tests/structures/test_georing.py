
from datetime import datetime

from pytest import approx

from geostructures import GeoBox, GeoCircle, GeoPolygon, GeoRing
from geostructures.coordinates import Coordinate
from geostructures.geodesic import destination_point
from geostructures.utils.functions import round_half_up

from tests.functions import (
    assert_coordinates_equal, assert_geopolygons_equal, default_test_datetime,
    geojson_round_trip, shapely_round_trip,
)


def test_georing_eq():
    wedge = GeoRing(Coordinate(0.0, 0.0), 500, 1000, 90, 180, dt=default_test_datetime)

    w2 = GeoRing(Coordinate(0.0, 0.0), 500, 1000, 90, 180, dt=default_test_datetime)
    assert w2 == wedge

    w2 = GeoRing(Coordinate(1.0, 0.0), 500, 1000, 90, 180, dt=default_test_datetime)
    assert w2 != wedge

    w2 = GeoRing(Coordinate(1.0, 0.0), 600, 1000, 90, 180, dt=default_test_datetime)
    assert w2 != wedge

    w2 = GeoRing(Coordinate(1.0, 0.0), 500, 2000, 90, 180, dt=default_test_datetime)
    assert w2 != wedge

    w2 = GeoRing(Coordinate(1.0, 0.0), 500, 1000, 80, 180, dt=default_test_datetime)
    assert w2 != wedge

    w2 = GeoRing(Coordinate(1.0, 0.0), 500, 1000, 90, 190, dt=default_test_datetime)
    assert w2 != wedge

    w2 = GeoRing(Coordinate(0.0, 0.0), 500, 1000, 90, 180, dt=datetime(1970, 1, 1, 1, 1))
    assert w2 != wedge

    assert 'test' != wedge


def test_georing_hash():
    wedge = GeoRing(Coordinate(0.0, 0.0), 500, 1000, 90, 180, dt=default_test_datetime)
    w2 = GeoRing(Coordinate(0.0, 0.0), 500, 1000, 90, 180, dt=default_test_datetime)
    assert len({wedge, w2}) == 1


def test_georing_repr():
    wedge = GeoRing(Coordinate(0.0, 0.0), 500, 1000, 90, 180)
    assert repr(wedge) == '<GeoRing at (0.0, 0.0); radii 500.0/1000.0; 90.0-180.0 degrees>'

    ring = GeoRing(Coordinate(0.0, 0.0), 500, 1000)
    assert repr(ring) == '<GeoRing at (0.0, 0.0); radii 500.0/1000.0>'


def test_georing_bounds():
    ring = GeoRing(Coordinate(0., 0.), 1000, 5000)
    expected = (-0.0449661, -0.0449661, 0.0449661, 0.0449661)
    for b1, b2 in zip(ring.bounds, expected):
        assert b1 == approx(b2, abs=1e-6)

    wedge = GeoRing(Coordinate(0., 0.), 1000, 5000, 90, 180)
    expected = (0., -0.0449661, 0.0449661, 0.)
    for b1, b2 in zip(wedge.bounds, expected):
        assert b1 == approx(b2, abs=1e-6)


def test_georing_bounding_coords():
    wedge = GeoRing(Coordinate(0.0, 0.0), 500, 1000, 90, 180)
    expected = [
        Coordinate(0.0, -0.0089932),
        Coordinate(0.0014068, -0.0088825),
        Coordinate(0.0027791, -0.0085531),
        Coordinate(0.0040828, -0.008013),
        Coordinate(0.0052861, -0.0072757),
    ]
    for c1, c2 in zip(wedge.bounding_coords(), expected):
        assert_coordinates_equal(c1, c2)

    # Assert self-closing
    assert_coordinates_equal(wedge.bounding_coords()[0], wedge.bounding_coords()[-1])

    ring = GeoRing(Coordinate(0.0, 0.0), 500, 1000)
    expected = [
        Coordinate(-0.0, 0.0089932),
        Coordinate(-0.0015617, 0.0088566),
        Coordinate(-0.0030759, 0.0084509),
        Coordinate(-0.0044966, 0.0077884),
        Coordinate(-0.0057807, 0.0068892),
    ]
    for c1, c2 in zip(ring.bounding_coords(), expected):
        assert_coordinates_equal(c1, c2)

    # assert self-closing
    assert_coordinates_equal(ring.bounding_coords()[0], ring.bounding_coords()[-1])


def test_georing_centroid():
    ring = GeoRing(Coordinate(0.0, 0.0), 500, 1000)
    assert ring.centroid == Coordinate(0.0, 0.0)

    wedge = GeoRing(Coordinate(0.0, 0.0), 500, 1000, 90, 180)
    received = Coordinate(
        round_half_up(wedge.centroid.longitude, 8),
        round_half_up(wedge.centroid.latitude, 8),
    )
    assert received == Coordinate(0.00444382, -0.00444382)


def test_georing_contains():
    ring = GeoRing(Coordinate(0.0, 0.0), 500, 1000)
    wedge = GeoRing(Coordinate(0.0, 0.0), 500, 1000, 90, 180)
    # 750 meters east
    assert ring.contains_coordinate(Coordinate(0.0067449, 0.0))
    assert wedge.contains_coordinate(Coordinate(0.0067449, 0.0))

    # 750 meters west (outside wedge angle)
    assert ring.contains_coordinate(Coordinate(-0.0067449, 0.0))
    assert not wedge.contains_coordinate(Coordinate(-0.0067449, 0.0))

    # Centerpoint (not in shape)
    assert not ring.contains_coordinate(Coordinate(0.0, 0.0))
    assert not wedge.contains_coordinate(Coordinate(0.0, 0.0))

    # Along edge (1000m east)
    assert ring.contains_coordinate(Coordinate(0.0089932, 0.0))
    assert wedge.contains_coordinate(Coordinate(0.0089932, 0.0))

    ring = GeoRing(Coordinate(0.0, 0.0), 500, 1000, holes=[GeoCircle(Coordinate(0.0067449, 0.0), 200)])
    assert not ring.contains_coordinate(Coordinate(0.0067449, 0.0))


def test_georing_wedge_with_zero_angle_min():
    # angle_min=0 used to be treated as falsy, i.e. "not a wedge"
    wedge = GeoRing(Coordinate(0., 0.), 1_000, 2_000, angle_min=0., angle_max=90.)
    assert wedge.centroid != Coordinate(0., 0.)
    assert '0.0-90.0 degrees' in repr(wedge)

    full_ring = GeoRing(Coordinate(0., 0.), 1_000, 2_000)
    assert 'degrees' not in repr(full_ring)
    assert full_ring.centroid == Coordinate(0., 0.)


def test_georing_wedge_spanning_north():
    # Bearings must be compared modularly for wedges that span due north
    wedge = GeoRing(Coordinate(0., 0.), 1_000, 2_000, angle_min=315., angle_max=405.)

    # ~1.5km due north (bearing 0)
    assert wedge.contains_coordinate(Coordinate(0., 0.0135))
    # ~1.5km due south (bearing 180)
    assert not wedge.contains_coordinate(Coordinate(0., -0.0135))


def test_georing_copy():
    ring = GeoRing(Coordinate(0., 1.), 500, 200)
    ring_copy = ring.copy()

    # Assert equality but different pointer
    assert ring == ring_copy
    assert ring is not ring_copy


def test_georing_serialization_round_trips():
    # Rings parse back as polygons with a hole; wedges as simple polygons
    ring = GeoRing(Coordinate(0.0, 0.0), 500, 1000, dt=default_test_datetime)
    geojson_round_trip(ring)
    shapely_round_trip(ring)

    wedge = GeoRing(Coordinate(0.0, 0.0), 500, 1000, 90, 180, dt=default_test_datetime)
    geojson_round_trip(wedge)
    shapely_round_trip(wedge)


def test_georing_to_wkt():
    # Full Ring: a polygon with the inner circle as a hole
    ring = GeoRing(Coordinate(0.0, 0.0), 500, 1000)
    parsed_poly = GeoPolygon.from_wkt(ring.to_wkt())

    expected_outer = GeoCircle(Coordinate(0.0, 0.0), 1000).bounding_coords()
    expected_inner = GeoCircle(Coordinate(0.0, 0.0), 500).bounding_coords()
    expected_poly = GeoPolygon(expected_outer, holes=[GeoPolygon(expected_inner)])

    assert_geopolygons_equal(parsed_poly, expected_poly)

    # Wedge: just the standard polygon representation
    wedge = GeoRing(Coordinate(0.0, 0.0), 500, 1000, 90, 180)
    parsed_wedge = GeoPolygon.from_wkt(wedge.to_wkt())
    assert_geopolygons_equal(parsed_wedge, wedge.to_polygon())


def test_georing_linear_rings():
    ring = GeoRing(Coordinate(0.0, 0.0), 500, 1000)
    rings = ring.linear_rings()
    assert len(rings) == 2  # should have outer and inner shell

    expected = [
        Coordinate(-0.0, 0.0089932),
        Coordinate(-0.0015617, 0.0088566),
        Coordinate(-0.0030759, 0.0084509),
        Coordinate(-0.0044966, 0.0077884),
        Coordinate(-0.0057807, 0.0068892)
    ]
    for c1, c2 in zip(rings[0][:5], expected):
        assert_coordinates_equal(c1, c2)

    # Assert self-closing
    assert_coordinates_equal(rings[0][0], rings[0][-1])
    assert_coordinates_equal(rings[1][0], rings[1][-1])

    wedge = GeoRing(Coordinate(0.0, 0.0), 500, 1000, 90, 180)
    rings = wedge.linear_rings()
    assert len(rings) == 1
    assert_coordinates_equal(rings[0][0], rings[0][-1])


def test_georing_to_polygon():
    ring = GeoRing(Coordinate(0.0, 0.0), 500, 1000, dt=default_test_datetime)

    # Because its not a wedge, should become a polygon with a hole
    rings = ring.linear_rings()
    assert ring.to_polygon() == GeoPolygon(
        rings[0],
        holes=[GeoPolygon(rings[1])],
        dt=default_test_datetime
    )

    # Is a wedge, so just a normal polygon
    wedge = GeoRing(Coordinate(0.0, 0.0), 500, 1000, 90, 180, dt=default_test_datetime)
    rings = wedge.linear_rings()
    assert wedge.to_polygon() == GeoPolygon(
        rings[0],
        dt=default_test_datetime
    )


def test_georing_to_polygon_does_not_mutate():
    # to_polygon() used to alias and extend self.holes, compounding on each call
    ring = GeoRing(Coordinate(0., 0.), 1_000, 2_000)
    assert ring.holes == []

    poly1 = ring.to_polygon()
    assert ring.holes == []
    assert len(poly1.holes) == 1  # The inner ring

    poly2 = ring.to_polygon()
    assert len(poly2.holes) == 1

    # Explicit holes should appear exactly once, alongside the inner ring
    ring = GeoRing(
        Coordinate(0., 0.), 1_000, 2_000,
        holes=[GeoCircle(Coordinate(0.015, 0.), 100)]
    )
    assert len(ring.to_polygon().holes) == 2
    assert len(ring.to_polygon().holes) == 2
    assert len(ring.holes) == 1


def test_georing_circumscribing_rectangle():
    ring = GeoRing(Coordinate(0.0, 0.0), 500, 1000, dt=default_test_datetime)
    wedge = GeoRing(Coordinate(0.0, 0.0), 500, 1000, 90, 180, dt=default_test_datetime)

    max_lon, _ = destination_point(ring.centroid, 90, 1000).to_float()
    min_lon, _ = destination_point(ring.centroid, -90, 1000).to_float()
    _, max_lat = destination_point(ring.centroid, 0, 1000).to_float()
    _, min_lat = destination_point(ring.centroid, 180, 1000).to_float()

    actual = ring.circumscribing_rectangle()
    expected = GeoBox(
        Coordinate(min_lon, max_lat),
        Coordinate(max_lon, min_lat),
        dt=default_test_datetime
    )
    assert_coordinates_equal(actual.nw_bound, expected.nw_bound)
    assert_coordinates_equal(actual.se_bound, expected.se_bound)

    actual = wedge.circumscribing_rectangle()
    expected = GeoBox(
        wedge.center,
        Coordinate(max_lon, min_lat),
        dt=default_test_datetime
    )
    assert_coordinates_equal(actual.nw_bound, expected.nw_bound)
    assert_coordinates_equal(actual.se_bound, expected.se_bound)


def test_georing_circumscribing_circle():
    ring = GeoRing(Coordinate(0.0, 0.0), 500, 1000)
    actual = ring.circumscribing_circle()
    expected = GeoCircle(
        ring.centroid,
        ring.outer_radius
    )
    assert_coordinates_equal(actual.centroid, expected.centroid)
    assert actual.radius == approx(expected.radius, abs=1e-6)

    wedge = GeoRing(Coordinate(0.0, 0.0), 500, 1000, 90, 180)
    circ = wedge.circumscribing_circle()
    received_centroid = Coordinate(
        round_half_up(circ.centroid.longitude, 8),
        round_half_up(circ.centroid.latitude, 8),
    )
    assert_coordinates_equal(received_centroid, Coordinate(0.00444382, -0.00444382))
    assert circ.radius == approx(707.1555054, abs=1e-6)

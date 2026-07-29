
import itertools
import math
import random
import sys

from pytest import approx
import pytest

from geostructures._const import COORDINATE_TOLERANCE, EARTH_RADIUS_METERS
from geostructures._geometry import *
from geostructures._geometry import _circumcircle, _diameter_circle
from geostructures.coordinates import Coordinate
from geostructures.utils.functions import round_half_up

from tests.functions import assert_coordinates_equal


def test_coordinate_vector_cross_product():
    assert coordinate_vector_cross_product(Coordinate(0., 0.), Coordinate(0., 1.), Coordinate(1., 0.)) == -1.
    assert coordinate_vector_cross_product(Coordinate(0., 0.), Coordinate(1., 0.), Coordinate(0., 1.)) == 1.
    assert coordinate_vector_cross_product(Coordinate(0., 0.), Coordinate(-1., 0.), Coordinate(1., 0.)) == 0.


def test_convex_hull():
    coords = [
        Coordinate(0., 0.), Coordinate(1., 0.),
        Coordinate(0., 1.), Coordinate(1., 1.),
        Coordinate(0.5, 0.5)
    ]
    assert convex_hull(coords) == [
        Coordinate(0., 0.), Coordinate(1., 0.), Coordinate(1., 1.), Coordinate(0., 1.),
        Coordinate(0., 0.)
    ]

    assert convex_hull([Coordinate(0., 0.)]) == [Coordinate(0., 0.)]


def test_minimum_bounding_circle():
    points = [
        Coordinate(0, 5),
        Coordinate(0, 0),
        Coordinate(2, 1),
        Coordinate(4, 3)
    ]
    cc = minimum_bounding_circle(points)
    assert round_half_up(cc[0].latitude, 6) == 2.499407
    assert round_half_up(cc[0].longitude, 6) == 1.248383
    assert round_half_up(cc[1], 0) == 310640

    with pytest.raises(ValueError):
        minimum_bounding_circle([])


def test_minimum_bounding_circle_encloses_and_minimizes():
    # Every circle produced has to contain every point, and be no larger than the
    # smallest circle through any pair or triple of them
    random.seed(0)
    for _ in range(50):
        points = [
            Coordinate(random.uniform(-8, 8), random.uniform(-8, 8))
            for _ in range(random.randint(2, 8))
        ]
        center, radius = minimum_bounding_circle(points)
        assert max(dist_xyz_meters(center, point) for point in points) <= radius + 1e-3

        best = min(
            (
                circle[1] for circle in (
                    [_diameter_circle(*pair) for pair in itertools.combinations(points, 2)]
                    + [_circumcircle(*trio) for trio in itertools.combinations(points, 3)]
                )
                if circle is not None
                and max(dist_xyz_meters(circle[0], p) for p in points) <= circle[1] + 1e-3
            ),
            default=None
        )
        assert best is None or radius <= best + 1e-3


def test_minimum_bounding_circle_degenerate_inputs():
    # A single coordinate, and any number of copies of it, is its own circle
    assert minimum_bounding_circle([Coordinate(3., 4.)]) == (Coordinate(3., 4.), 0.0)
    assert minimum_bounding_circle([Coordinate(1., 1.)] * 6) == (Coordinate(1., 1.), 0.0)

    # Collinear coordinates are spanned by their two extremes
    center, radius = minimum_bounding_circle([Coordinate(float(i), 0.) for i in range(6)])
    assert_coordinates_equal(center, Coordinate(2.5, 0.))
    assert radius == approx(dist_xyz_meters(Coordinate(0., 0.), Coordinate(2.5, 0.)))


def test_minimum_bounding_circle_antimeridian_and_poles():
    # The spherical formulation has to handle shapes a planar one would tear apart
    for points, expected_center in (
        (
            [
                Coordinate(179., 0.), Coordinate(-179., 0.),
                Coordinate(179.5, 1.), Coordinate(-179.5, -1.),
            ],
            Coordinate(-180., 0.),
        ),
        (
            [
                Coordinate(0., 89.5), Coordinate(90., 89.5),
                Coordinate(180., 89.5), Coordinate(-90., 89.5),
            ],
            Coordinate(-90., 90.),
        ),
    ):
        center, radius = minimum_bounding_circle(points)
        assert_coordinates_equal(center, expected_center)
        assert max(dist_xyz_meters(center, point) for point in points) <= radius + 1e-3


def test_minimum_bounding_circle_beyond_a_hemisphere():
    # No 90 degree cap can cover these, so the result falls back to a circle that still
    # encloses every coordinate but is not the smallest one
    points = [
        Coordinate(-170., -60.), Coordinate(170., -60.),
        Coordinate(0., 60.), Coordinate(90., 10.),
    ]
    center, radius = minimum_bounding_circle(points)
    assert max(dist_xyz_meters(center, point) for point in points) <= radius + 1e-3


def test_circumcircle_degenerate():
    # A repeated coordinate collapses the circumcenter formula's numerator to zero
    assert _circumcircle(Coordinate(1., 1.), Coordinate(1., 1.), Coordinate(2., 2.)) is None

    # Coordinates sharing a great circle are not degenerate - that great circle is their
    # circumcircle, centered on its pole with a quarter-circumference radius
    center, radius = _circumcircle(Coordinate(0., 0.), Coordinate(180., 0.), Coordinate(90., 0.))
    assert abs(center.latitude) == approx(90.)
    assert radius == approx(math.pi / 2 * EARTH_RADIUS_METERS)


def test_minimum_bounding_circle_is_deterministic():
    # The incremental construction shuffles its input, but from a private generator
    # seeded per call - identical input has to produce an identical circle, and the
    # caller's own random stream must be left alone
    random.seed(9)
    points = [Coordinate(random.uniform(-5, 5), random.uniform(-5, 5)) for _ in range(30)]
    assert len({minimum_bounding_circle(points) for _ in range(20)}) == 1

    # Cocircular points, where any of several circles could win a noisy tie
    square = [Coordinate(0., 0.), Coordinate(1., 0.), Coordinate(1., 1.), Coordinate(0., 1.)]
    assert len({minimum_bounding_circle(square) for _ in range(20)}) == 1

    random.seed(1234)
    expected = [random.random() for _ in range(3)]
    random.seed(1234)
    minimum_bounding_circle(points)
    assert [random.random() for _ in range(3)] == expected


def test_minimum_bounding_circle_does_not_recurse():
    # The previous implementation recursed once per coordinate and raised a
    # RecursionError somewhere north of 990 of them
    random.seed(3)
    points = [
        Coordinate(random.uniform(-10, 10), random.uniform(-10, 10))
        for _ in range(sys.getrecursionlimit() * 3)
    ]
    center, radius = minimum_bounding_circle(points)
    assert max(dist_xyz_meters(center, point) for point in points) <= radius + 1e-3


def test_dist_xyz_meters():
    # Sourced from haversine package
    actual_dist_meters = 157.25359
    calc_dist = dist_xyz_meters(Coordinate(0.0, 0.0), Coordinate(0.001, 0.001))
    assert round(actual_dist_meters) == round(calc_dist)

    actual_dist_meters = 157_249.59847
    calc_dist = dist_xyz_meters(Coordinate(0.0, 0.0), Coordinate(1.0, 1.0))
    assert abs(round(actual_dist_meters) - round(calc_dist)) < 2

    # Antimeridian test
    actual_dist_meters = 222390
    calc_dist = dist_xyz_meters(Coordinate(179., 0.), Coordinate(-179., 0.))
    assert round(calc_dist) == actual_dist_meters


def test_do_bounds_overlap():
    assert do_bounds_overlap(
        (0., 1.),
        (0.5, 1.5)
    )

    assert do_bounds_overlap(
        (0., 1.),
        (1., 2)
    )

    assert not do_bounds_overlap(
        (0., 1.),
        (2., 3)
    )


def test_do_edges_intersect():
    edge_a = [
        (Coordinate(179, -1), Coordinate(-179, 1)),
    ]
    edge_b = [
        (Coordinate(178, -1), Coordinate(-178, 1))
    ]
    assert do_edges_intersect(edge_a, edge_b)


def test_do_edges_intersect_shared_endpoint():
    """
    A segment whose highest point is another segment's lowest point touches it, and
    both argument orders have to agree. Without start-before-stop tie ordering the
    first segment is retired from the active set before the second is tested.
    """
    edge_a = [(Coordinate(0., 0.), Coordinate(0., 5.))]
    edge_b = [(Coordinate(0., 5.), Coordinate(5., 10.))]

    assert do_edges_intersect(edge_a, edge_b)
    assert do_edges_intersect(edge_b, edge_a)


def test_do_edges_intersect_duplicate_segments():
    """
    A group may contain the same segment twice - a linestring that doubles back, or a
    polygon that retraces an edge. Both copies need their own start/stop pair.
    """
    # Both edges normalize to the same (lesser latitude first) tuple
    edge_a = [
        (Coordinate(20., 20.), Coordinate(30., 30.)),
        (Coordinate(30., 30.), Coordinate(20., 20.)),
    ]
    edge_b = [(Coordinate(0., 0.), Coordinate(1., 1.))]
    assert not do_edges_intersect(edge_a, edge_b)

    edge_b = [(Coordinate(30., 20.), Coordinate(20., 30.))]
    assert do_edges_intersect(edge_a, edge_b)


def test_do_edges_intersect_horizontal_edge():
    """A horizontal edge produces two events at the same sweep position, and must
    still leave the active set once passed."""
    edge_a = [(Coordinate(0., 0.), Coordinate(10., 0.))]
    edge_b = [(Coordinate(5., 50.), Coordinate(5., 60.))]
    assert not do_edges_intersect(edge_a, edge_b)

    edge_b = [(Coordinate(5., -5.), Coordinate(5., 5.))]
    assert do_edges_intersect(edge_a, edge_b)


def test_ensure_edge_bounds():
    assert ensure_edge_bounds(Coordinate(179., 0.), Coordinate(179.5, 0.)) == (Coordinate(179., 0.), Coordinate(179.5, 0.))
    assert ensure_edge_bounds(Coordinate(179., 0.), Coordinate(-179, 0.)) == (Coordinate(179., 0.), Coordinate(181, 0., _bounded=False))
    assert ensure_edge_bounds(Coordinate(-179., 0.), Coordinate(179, 0.)) == (Coordinate(-179., 0.), Coordinate(-181, 0., _bounded=False))


def test_find_line_intersection():
    # X-shape, intersect in middle
    line1 = (Coordinate(0.0, 1.0), Coordinate(1.0, 0.0))
    line2 = (Coordinate(0.0, 0.0), Coordinate(1.0, 1.0))
    assert find_line_intersection(line1, line2) == (Coordinate(0.5, 0.5), False)

    # Coordinates run right to left, should get flipped
    line1 = (Coordinate(1.0, 0.0), Coordinate(0.0, 1.0), )
    line2 = (Coordinate(1.0, 1.0), Coordinate(0.0, 0.0), )
    assert find_line_intersection(line1, line2) == (Coordinate(0.5, 0.5), False)

    # Intersect at ends - boundary intersection
    line1 = (Coordinate(0.0, 1.0), Coordinate(1.0, 0.0))
    line2 = (Coordinate(1.0, 0.0), Coordinate(2.0, 0.0))
    assert find_line_intersection(line1, line2) == (Coordinate(1.0, 0.0), True)

    # Parallel lines
    line1 = (Coordinate(0.0, 1.0), Coordinate(1.0, 0.0))
    line2 = (Coordinate(1.0, 2.0), Coordinate(2.0, 1.0))
    assert not find_line_intersection(line1, line2)

    # Out of bounds
    line1 = (Coordinate(5.0, 4.0), Coordinate(5.0, 4.0))
    line2 = (Coordinate(1.0, 2.0), Coordinate(2.0, 1.0))
    assert not find_line_intersection(line1, line2)


def test_find_line_intersection_boundary_with_z():
    # Regression: the boundary flag compared a (lon, lat) pair against whole to_float()
    # tuples, which carry Z when present - so a Z-carrying endpoint never matched
    line1 = (Coordinate(0.0, 1.0, z=5.), Coordinate(1.0, 0.0, z=5.))
    line2 = (Coordinate(1.0, 0.0, z=7.), Coordinate(2.0, 0.0, z=7.))
    coord, is_boundary = find_line_intersection(line1, line2)
    assert_coordinates_equal(coord, Coordinate(1.0, 0.0))
    assert is_boundary


def test_find_line_intersection_tolerance():
    # An endpoint touch that misses by less than the coordinate tolerance still counts
    # as a boundary intersection, rather than depending on where the values happen to
    # sit relative to a rounding grid
    line1 = (Coordinate(0.0, 1.0), Coordinate(1.0, 0.0))
    line2 = (Coordinate(1.0 - COORDINATE_TOLERANCE / 10, 0.0), Coordinate(2.0, 0.0))
    result = find_line_intersection(line1, line2)
    assert result is not None
    assert result[1]

    # A gap comfortably wider than the tolerance does not
    line2 = (Coordinate(1.0 + COORDINATE_TOLERANCE * 100, 0.0), Coordinate(2.0, 0.0))
    result = find_line_intersection(line1, line2)
    assert result is None or not result[1]

    # In bounds and not parallel but no intersection
    line1 = (Coordinate(0.0, 0.0), Coordinate(0.5, 0.5))
    line2 = (Coordinate(0.0, 0.5), Coordinate(0.1, 0.3))
    assert not find_line_intersection(line1, line2)

    # Antimeridian test
    line1 = (Coordinate(179, -1), Coordinate(-179, 1))
    line2 = (Coordinate(178, -1), Coordinate(-178, 1))
    assert find_line_intersection(line1, line2) == (Coordinate(-180., -0.), False)


def test_is_point_in_line():
    assert is_point_in_line(
        Coordinate(0.5, 0.5),
        (Coordinate(0., 0.), Coordinate(1., 1.))
    )

    assert not is_point_in_line(
        Coordinate(0.5, 0.6),
        (Coordinate(0., 0.), Coordinate(1., 1.))
    )

    assert not is_point_in_line(
        Coordinate(1.5, 0.6),
        (Coordinate(0., 0.), Coordinate(1., 1.))
    )


def test_is_counter_clockwise():
    # Counter clockwise box
    assert is_counter_clockwise([
        Coordinate(0., 0.), Coordinate(1., 0.), Coordinate(1., 1.),
        Coordinate(0., 1.), Coordinate(0., 0.)
    ])

    # Same box, reversed to be clockwise
    assert not is_counter_clockwise(list(reversed([
        Coordinate(0., 0.), Coordinate(1., 0.), Coordinate(1., 1.),
        Coordinate(0., 1.), Coordinate(0., 0.)
    ])))

    # Figure 8
    assert is_counter_clockwise([
        Coordinate(0.5, 1.), Coordinate(0.25, 0.75), Coordinate(0.5, 0.5),
        Coordinate(0.75, 0.25), Coordinate(0.5, 0.0), Coordinate(0.25, 0.25),
        Coordinate(0.5, 0.5), Coordinate(0.75, 0.75), Coordinate(0.5, 1.)
    ])


def test_convert_trig_angle():
    assert convert_trig_angle(90) == 0.
    assert convert_trig_angle(30) == 60.
    assert convert_trig_angle(0.) == 90.
    assert convert_trig_angle(-90) == 180.

    assert convert_trig_angle(-450) == 180.
    assert convert_trig_angle(450) == 0.

    assert convert_trig_angle(180) == 270.
    assert convert_trig_angle(270) == 180.


def test_dist_xyz_meters_identical_coordinates():
    # Taking acos of a near-1.0 dot product used to raise a math domain error here, and
    # then, once clamped, reported a tenth of a meter between a coordinate and itself
    for lon in range(-180, 180, 7):
        for lat in range(-89, 90, 7):
            coord1 = Coordinate(lon + 0.123456789, lat + 0.987654321)
            coord2 = Coordinate(lon + 0.123456789, lat + 0.987654321)
            assert dist_xyz_meters(coord1, coord2) == 0.


def test_dist_xyz_meters_short_distances():
    # Precision at short range matters because the bounding circle compares a radius
    # against these distances; a meter's worth of noise mis-ranks candidate circles
    for meters in (0.01, 1., 100.):
        degrees = math.degrees(meters / EARTH_RADIUS_METERS)
        calculated = dist_xyz_meters(Coordinate(0., 0.), Coordinate(0., degrees))
        assert calculated == approx(meters, rel=1e-9)


def test_minimum_bounding_circle_obtuse_triangle():
    # For an obtuse triangle the smallest circle is centered at the midpoint of the
    # longest side, rather than at the circumcenter
    center, radius = minimum_bounding_circle([
        Coordinate(0., 0.), Coordinate(1., 0.), Coordinate(0.9, 0.01)
    ])
    assert abs(center.longitude - 0.5) < 0.01
    assert abs(center.latitude - 0.005) < 0.01
    assert radius > 0

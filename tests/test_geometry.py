
from geostructures._geometry import *
from geostructures.coordinates import Coordinate
from geostructures.utils.functions import round_half_up


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


def test_circumscribing_circle_for_polygon():
    points = [
        Coordinate(0, 5),
        Coordinate(0, 0),
        Coordinate(2, 1),
        Coordinate(4, 3)
    ]
    cc = circumscribing_circle_for_polygon(points, [])
    assert round_half_up(cc[0].latitude, 6) == 2.499407
    assert round_half_up(cc[0].longitude, 6) == 1.248383
    assert round_half_up(cc[1], 0) == 310640


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

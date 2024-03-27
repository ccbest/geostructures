
import math

from geostructures.calc import *
from geostructures.coordinates import Coordinate
from geostructures.utils.functions import round_half_up


def test_bearing_degrees():
    assert bearing_degrees(Coordinate(0.0, 0.0), Coordinate(0.001, 0.001)) == 45.
    assert bearing_degrees(Coordinate(0.0, 0.0), Coordinate(0.001, 0.001), precision=9) == 44.999999996

def test_circumscribing_circle_for_polygon():
    points =[
        Coordinate(0,5),
        Coordinate(0,0),
        Coordinate(2,1),
        Coordinate(4,3)
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
    assert rotate_coordinates(points, Coordinate(0.0, 0.0), 45, precision=3) == [
        Coordinate(0.707, 0.707),
        Coordinate(0.707, 0.707),
        Coordinate(0.707, 0.707),
    ]

    # Antimeridian test
    points = [
        Coordinate(-179, 0.),
        Coordinate(179, 0.)
    ]
    assert rotate_coordinates(points, Coordinate(179.999, 0.), 135) == [
        Coordinate(179.2911861, 0.7078139),
        Coordinate(-179.2946003, -0.7063997)
    ]

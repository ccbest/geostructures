""" Geometric calculations for Coordinates and Geostructures """

__all__ = [
    'bearing_degrees', 'dist_xyz_meters', 'circumscribing_circle_for_polygon',
    'ensure_edge_bounds', 'haversine_distance_meters', 'inverse_haversine_degrees',
    'inverse_haversine_radians', 'rotate_coordinates', 'find_line_intersection',
    'do_edges_intersect'
]

import math
import random
from typing import List, Optional, Set, Tuple

import numpy as np
from numpy.linalg import norm

from geostructures.coordinates import Coordinate
from geostructures.utils.functions import round_half_up


_EARTH_RADIUS = 6_371_000  # meters - WGS84


def _circumscribing_circle_for_triangle(
    points: List[Coordinate]
) -> Tuple[Optional[Coordinate], Optional[float]]:
    """
    Supporting function for circumscribing_circle_for_polygon().

    Can be called with up to three points to return the center and radius of the
    circumscribing circle.

    Zero points returns None center and radius (will fail check in calling function).

    One point returns itself as center and zero radius (will also fail check).

    Two points returns the midpoint as center and half the distance as radius.

    Three points checks every point pair as a possible diameter for the circle. If no
    pair qualifies, uses the circumcenter formula from
    https://brsr.github.io/2021/05/02/spherical-triangle-centers.html

    Args:
        points:
            A list of Coordinates. Will error if more than three.

    Returns:
        (Coordinate, float) tuple of (Circumcenter, Radius in meters)

    """
    assert len(points) <= 3
    if len(points) == 0:
        return (None, None)
    if len(points) == 1:
        return (points[0], 0.0)
    if len(points) == 2:
        midp = [(v1+v2)/2 for v1, v2 in zip(points[0].xyz, points[1].xyz)]
        midp_norm = norm(midp)
        midp_coord = Coordinate._from_xyz([i/midp_norm for i in midp])
        rad = dist_xyz_meters(midp_coord, points[0])
        return (midp_coord, rad)

    if not _test_counter_clockwise(points):
        points = points[::-1]

    # Test for trivial circle
    for i in range(3):
        p = points[i]
        other_p = points[:i] + points[i+1:]
        midp = [(v1+v2)/2 for v1, v2 in zip(other_p[0].xyz, other_p[1].xyz)]
        midp_norm = norm(midp)
        midp_coord = Coordinate._from_xyz([i/midp_norm for i in midp])
        rad = dist_xyz_meters(midp_coord, other_p[0])
        # If this is true, the midpoint of one side is the center
        # (i.e. any obtuse/right triangle) and we are done
        if rad >= dist_xyz_meters(midp_coord, p):
            return (midp_coord, rad)

    [a, b, c] = [p.xyz for p in points]
    cc_num = np.cross(a, b) + np.cross(b, c) + np.cross(c, a)
    cc_norm = norm(cc_num)
    ctr = Coordinate._from_xyz([i/cc_norm for i in cc_num])
    rad = math.acos(np.dot(a, np.cross(b, c))/cc_norm) * _EARTH_RADIUS
    return (ctr, rad)


def _test_counter_clockwise(bounds: List[Coordinate]) -> bool:
    """
    Tests a polygon to determine whether it's defined in a counterclockwise
    (or mostly, for complex shapes) order.

    Args:
        bounds:
            A list of Coordinates, in order

    Returns:
        bool
    """
    ans = sum(
        (y.longitude - x.longitude) * (y.latitude + x.latitude)
        for x, y in map(
            lambda x: ensure_edge_bounds(x[0], x[1]),
            zip(bounds, [*bounds[1:], bounds[0]])
        )
    )
    return ans <= 0


def bearing_degrees(coord1: Coordinate, coord2: Coordinate, **kwargs) -> float:
    """
    Calculate the bearing in degrees between lon1/lat1 and lon2/lat2

    Args:
        coord1:
            The start point Coordinate

        coord2:
            The finish point Coordinate

    Keyword Args:
        precision: (int) (Default 5)
            The decimal precision to round the resulting bearing to

    Returns:
        (float) the bearing in degrees
    """
    d_lon = coord2.longitude - coord1.longitude
    x_val = math.cos(math.radians(coord2.latitude)) * math.sin(
        math.radians(d_lon)
    )
    y_val = math.cos(math.radians(coord1.latitude)) * math.sin(
        math.radians(coord2.latitude)
    ) - math.sin(math.radians(coord1.latitude)) * math.cos(
        math.radians(coord2.latitude)
    ) * math.cos(
        math.radians(d_lon)
    )
    bearing = (math.degrees(math.atan2(x_val, y_val)) + 360) % 360
    return round_half_up(bearing, kwargs.get('precision', 5))


def circumscribing_circle_for_polygon(
    all_points: List[Coordinate],
    known_points: List[Coordinate]
) -> Tuple[Optional[Coordinate], Optional[float]]:
    """
    Implements Welzl's algorithm to determine the circumscribing circle
    for a set of points.

    Args:
        points:
            A list of Coordinates. Will error if more than three.
        known_points:
            A list of Coordinates. Must be initialized with empty list.
            Used by recursive calls after initialization.

    Returns:
        (Coordinate, float) tuple of (Circumcenter, Radius in meters)
    """
    if len(known_points) == 3:
        return _circumscribing_circle_for_triangle(known_points)
    if len(all_points) == 0:
        return _circumscribing_circle_for_triangle(known_points)
    i = random.randrange(0, len(all_points))
    p = all_points[i]
    other_p = all_points[:i] + all_points[i+1:]
    ctr, rad = circumscribing_circle_for_polygon(
        other_p,
        known_points.copy()
    )
    if rad is not None and rad >= dist_xyz_meters(p, ctr):
        return (ctr, rad)
    known_points.append(p)
    return circumscribing_circle_for_polygon(other_p, known_points.copy())


def dist_xyz_meters(coord1: Coordinate, coord2: Coordinate) -> float:
    """
    Great circle distance formula that works with the cached .xyz
    property. Faster than haversine_distance_meters if each Coordinate
    is used in distance calculations more than twice on average.

    Args:
        coord1:
            A coordinate

        coord2:
            A second coordinate

    Returns:
        (float) the distance in meters
    """
    return math.acos(sum([an*bn for an, bn in zip(coord1.xyz, coord2.xyz)])) * _EARTH_RADIUS


def do_edges_intersect(
    edges_a: List[Tuple[Coordinate, Coordinate]],
    edges_b: List[Tuple[Coordinate, Coordinate]],
) -> bool:
    """
    Tests whether two shape edges ever intersect. Uses the sweep line algorithm
    to minimize the number of intersections calculated.

    Args:
        edges_a:
            The list of edges (pairs of coordinates) from the first group/shape

        edges_b:
            The list of edges (pairs of coordinates) from the second group/shape

    Returns:
        bool
    """
    class _Event:
        """
        The start or stop of a segment. Each segment gets two events that hash to
        the same value so they can be added and removed from the set of active
        events.

        Additionally stores the group id of the segment, so that segments from the same
        group are not intersected.
        """
        def __init__(
                self,
                x: float,
                is_start: bool,
                segment: Tuple[Coordinate, Coordinate],
                group: str
        ):
            self.x = x
            self.group = group
            self.is_start = is_start
            self.segment = segment

        def __lt__(self, other):
            """Required for sorting events"""
            return self.x < other.x

        def __hash__(self):
            """Required for creating a set of events"""
            return hash((self.segment, self.group))

        def __eq__(self, other):
            """Required for creating a set of events"""
            return self.segment == other.segment and self.group == other.group

    def _create_events(edges, group):
        """Creates 2x events per edge from a list of edges. Ensures the lesser x
        value corresponds to the start event."""
        _events = []
        for edge in edges:
            if edge[0].latitude > edge[1].latitude:
                edge = (edge[1], edge[0])

            _events += [
                _Event(edge[0].latitude, edge[0].latitude <= edge[1].latitude, edge, group),
                _Event(edge[1].latitude, edge[1].latitude <= edge[0].latitude, edge, group)
            ]
        return _events

    edges_a = [ensure_edge_bounds(x, y) for x, y in edges_a]
    edges_b = [ensure_edge_bounds(x, y) for x, y in edges_b]

    events = _create_events(edges_a, 'a')
    events += _create_events(edges_b, 'b')

    events.sort()
    active_events: Set[_Event] = set()
    for event in events:
        if not event.is_start:
            active_events.remove(event)
            continue

        # All edges belong to same group
        if len(set(x.group for x in [*active_events, event])) <= 1:
            active_events.add(event)
            continue

        for active_event in active_events:
            if event.group == active_event.group:
                continue

            intersection = find_line_intersection(active_event.segment, event.segment)
            if intersection:
                return True

        active_events.add(event)

    return False


def ensure_edge_bounds(coord1: Coordinate, coord2: Coordinate) -> Tuple[Coordinate, Coordinate]:
    """
    Ensures edges (line segments from extending from point A to point B) are properly bounded,
    such that a line which crosses the antimeridian does not mathematically circumnavigate
    the globe.

    If the shortest path between the start and end points of an edge crosses the antimeridian,
    this function will "unbound" the end point coordinate such that its latitude value
    is not limited to [-180, 180), thereby allowing the meridian crossing to be calculated
    correctly.

    Args:
        coord1:
            A geostructures coordinate representing the edge start

        coord2:
            A geostructures coordinate representing the edge finish

    Returns:
        A tuple of unbounded coordinates
    """
    if abs(coord1.longitude - coord2.longitude) > 180:
        adjusted_lon = coord2.longitude - 360 if coord1.longitude < 0 else coord2.longitude + 360
        return coord1, Coordinate(adjusted_lon, coord2.latitude, _bounded=False)
    return coord1, coord2


def find_line_intersection(
        line1: Tuple[Coordinate, Coordinate],
        line2: Tuple[Coordinate, Coordinate]
) -> Optional[Tuple[Coordinate, bool]]:
    """
    Finds the point of intersection between two lines, each defined by two Coordinates.
    Each line's x and y values are bound between the Coordinate pairs.

    Parallel overlapping lines are not considered intersecting.

    Args:
        line1:
            A 2-tuple of two Coordinates

        line2:
            A second 2-tuple of two Coordinates

    Returns:
        If a point of intersection if found, returns a 2-tuple consisting of:
            - The Coordinate of the intersection location
            - A boolean "is_boundary" representing whether the intersection falls
              directly on one of the points of the lines
    """
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    def get_line_bounds(
            line: Tuple[Tuple[float, float], Tuple[float, float]]
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Organizes line bounding points into min/max x and y values"""
        x_sort, y_sort = sorted([line[0][0], line[1][0]]), sorted([line[0][1], line[1][1]])
        return (
            (round_half_up(x_sort[0], 10), round_half_up(x_sort[1], 10)),
            (round_half_up(y_sort[0], 10), round_half_up(y_sort[1], 10)),
        )

    def test_ranges_overlap(range1: Tuple[float, float], range2: Tuple[float, float]):
        """Test whether two ranges overlap"""
        return max([range1[0], range2[0]]) <= min([range1[1], range2[1]])

    # Adjust lines if they cross the antimeridian
    line1 = ensure_edge_bounds(line1[0], line1[1])
    line2 = ensure_edge_bounds(line2[0], line2[1])

    line1_flt = (line1[0].to_float(), line1[1].to_float())
    if line1_flt[1][0] < line1_flt[0][0]:  # Flip order such that lower x value is first
        line1_flt = (line1_flt[1], line1_flt[0])

    line2_flt = (line2[0].to_float(), line2[1].to_float())
    if line2_flt[1][0] < line2_flt[0][0]:
        line2_flt = (line2_flt[1], line2_flt[0])

    line1_bounds, line2_bounds = get_line_bounds(line1_flt), get_line_bounds(line2_flt)

    if not (
        test_ranges_overlap(line1_bounds[0], line2_bounds[0]) and
        test_ranges_overlap(line1_bounds[1], line2_bounds[1])
    ):
        # line bounds do not overlap
        return None

    line1_diff = (line1_flt[0][0] - line1_flt[1][0], line1_flt[0][1] - line1_flt[1][1])
    xdiff = (line1_diff[0], line2_flt[0][0] - line2_flt[1][0])
    ydiff = (line1_flt[0][1] - line1_flt[1][1], line2_flt[0][1] - line2_flt[1][1])
    div = det(xdiff, (line1_diff[1], line2_flt[0][1] - line2_flt[1][1]))
    if div == 0:
        # lines are parallel
        return None

    d = (det(*line1_flt), det(*line2_flt))
    x_intersection = round_half_up(det(d, xdiff) / div, 10)
    y_intersection = round_half_up(det(d, ydiff) / div, 10)

    if (
        line1_bounds[0][0] <= x_intersection <= line1_bounds[0][1] and
        line2_bounds[0][0] <= x_intersection <= line2_bounds[0][1] and
        line1_bounds[1][0] <= y_intersection <= line1_bounds[1][1] and
        line2_bounds[1][0] <= y_intersection <= line2_bounds[1][1]
    ):
        return (
            Coordinate(x_intersection, y_intersection),

            # If intersecting point is a line endpoint, is a boundary intersection
            (x_intersection, y_intersection) in (*line1_flt, *line2_flt)
        )

    return None


def haversine_distance_meters(coord1: Coordinate, coord2: Coordinate) -> float:
    """
    Calculate the Haversine distance in km between two points

    Args:
        coord1:
            A coordinate

        coord2:
            A second coordinate

    Returns:
        (float) the bearing in degrees
    """
    coord1, coord2 = ensure_edge_bounds(coord1, coord2)

    lon1, lat1 = math.radians(coord1.longitude), math.radians(coord1.latitude)
    lon2, lat2 = math.radians(coord2.longitude), math.radians(coord2.latitude)

    d_lat, d_long = lat2 - lat1, lon2 - lon1
    var1 = (math.sin(d_lat / 2) ** 2) + math.cos(lat1) * math.cos(lat2) * (
        math.sin(d_long / 2) ** 2
    )
    return _EARTH_RADIUS * 2 * math.atan2(math.sqrt(var1), math.sqrt(1 - var1))


def inverse_haversine_degrees(
    start: Coordinate, angle_degrees: float, distance_meters: float
):
    """
    Give a start location, a direction of travel (in degrees clockwise from North), and a
    distance of travel, returns the finish location.

    Convenience wrapper around inverse_haversine_radians that just converts degrees to radians.

    Args:
        start: (Coordinate)
            The starting location

        angle_degrees: (float)
            The angle of heading, in degrees

        distance_meters: (float)
            The amount of movement, in meters

    Returns:
        (Coordinate)
    """
    return inverse_haversine_radians(
        start, math.radians(angle_degrees), distance_meters
    )


def inverse_haversine_radians(
    start: Coordinate, angle_radians: float, distance_meters: float
) -> Coordinate:
    """
    Given a start location, a direction of travel in radians), and a distance of travel, returns
    the finish location.

    Args:
        start: (Coordinate)
            The starting location

        angle_radians: (float)
            The angle of heading, in radians

        distance_meters: (float)
            The amount of movement, in meters

    Returns:
        (Coordinate)
    """
    _rad = distance_meters / _EARTH_RADIUS

    x0 = start.longitude * math.pi / 180
    y0 = start.latitude * math.pi / 180

    final_lat = math.asin(
        math.sin(y0) * math.cos(_rad)
        + math.cos(y0) * math.sin(_rad) * math.cos(angle_radians)
    )
    final_lon = x0 + math.atan2(
        math.sin(angle_radians) * math.sin(_rad) * math.cos(y0),
        math.cos(_rad) - math.sin(y0) * math.sin(final_lat),
    )
    final_lat = round_half_up(final_lat * 180 / math.pi, 7)
    final_lon = round_half_up(final_lon * 180 / math.pi, 7)

    return Coordinate(final_lon, final_lat)


def rotate_coordinates(
        coords: List[Coordinate],
        origin: Coordinate,
        degrees: float,
        precision: int = 7,
) -> List[Coordinate]:
    """
    Rotate a set of coordinates around an origin. Returned Coordinate precision will be
    determined by the respective Coordinate's least significant precision.

    Args:
        coords:
            A list of Coordinates

        origin:
            The centerpoint around which to rotate other Coordinates

        degrees:
            The degrees of rotation to be applied

        precision: (Default 7)
            The decimal precision to round results to

    Returns:
        List[Coordinate]
    """
    coords = [ensure_edge_bounds(origin, x)[1] for x in coords]
    angle = np.deg2rad(degrees)
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    o = np.atleast_2d(origin.to_float())
    p = np.atleast_2d([x.to_float() for x in coords])
    return [
        Coordinate(
            round_half_up(coord[0], precision),
            round_half_up(coord[1], precision)
        )
        for coord in (R @ (p.T - o.T) + o.T).T
    ]

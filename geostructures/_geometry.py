"""
Internal module defining geometric functions used by geostructures
"""
import math
import random
from typing import List, Tuple, Optional, cast, Set

import numpy as np
from numpy.linalg import norm

from geostructures._const import EARTH_RADIUS
from geostructures.coordinates import Coordinate
from geostructures.utils.functions import round_half_up


def coordinate_vector_cross_product(o: Coordinate, a: Coordinate, b: Coordinate):
    """
    2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.

    Args:
        o: (Coordinate)
            The origin coordinate

        a: (Coordinate)
            The A coordinate, forming the OA vector

        b: (Coordinate)
            The B coordinate, forming the OB vector

    Returns:
        a positive value if OAB makes a counter-clockwise turn, negative for clockwise turn,
        and zero if the points are collinear.
    """
    return (
        (a.longitude - o.longitude) * (b.latitude - o.latitude) -
        (a.latitude - o.latitude) * (b.longitude - o.longitude)
    )


def convex_hull(coordinates: List[Coordinate]) -> List[Coordinate]:
    """
    Computes the convex hull of a set of coordinates using the Monotone Chain algorithm
    (aka Andrew's algorithm). O(n log n) complexity.

    Args:
        coordinates:
            A list of coordinates

    Returns:
        A linear ring representing the convex hull
    """

    # Sort the points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    coordinates = sorted(set(coordinates), key=lambda x: (x.longitude, x.latitude))

    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(coordinates) <= 1:
        return coordinates

    # Build lower hull
    lower: List[Coordinate] = []
    for coord in coordinates:
        while len(lower) >= 2 and coordinate_vector_cross_product(lower[-2], lower[-1], coord) <= 0:
            lower.pop()
        lower.append(coord)

    # Build upper hull
    upper: List[Coordinate] = []
    for coord in reversed(coordinates):
        while len(upper) >= 2 and coordinate_vector_cross_product(upper[-2], upper[-1], coord) <= 0:
            upper.pop()
        upper.append(coord)

    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of the lower list is omitted because it is repeated at
    # the beginning of the other list, but the last point of upper is
    # included because it closes the linear ring
    return lower[:-1] + upper


def circumscribing_circle_for_triangle(
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
        return None, None

    if len(points) == 1:
        return points[0], 0.0

    if len(points) == 2:
        midp = [(v1+v2)/2 for v1, v2 in zip(points[0].xyz, points[1].xyz)]
        midp_norm = norm(midp)
        midp_coord = Coordinate._from_xyz([i/midp_norm for i in midp])
        rad = dist_xyz_meters(midp_coord, points[0])
        return midp_coord, rad

    if not is_counter_clockwise(points):
        points = list(reversed(points))

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
            return midp_coord, rad

    [a, b, c] = [np.longdouble(p.xyz) for p in points]
    cc_num = np.cross(a, b) + np.cross(b, c) + np.cross(c, a)
    cc_norm = norm(cc_num)
    ctr = Coordinate._from_xyz([i/cc_norm for i in cc_num])
    rad = math.acos(np.dot(a, np.cross(b, c))/cc_norm) * EARTH_RADIUS
    return ctr, rad


def circumscribing_circle_for_polygon(
    all_points: List[Coordinate],
    known_points: List[Coordinate]
) -> Tuple[Optional[Coordinate], Optional[float]]:
    """
    Implements Welzl's algorithm to determine the circumscribing circle
    for a set of points.

    Args:
        all_points:
            A list of Coordinates. Will error if more than three.
        known_points:
            A list of Coordinates. Must be initialized with empty list.
            Used by recursive calls after initialization.

    Returns:
        (Coordinate, float) tuple of (Circumcenter, Radius in meters)
    """
    if len(known_points) == 3:
        return circumscribing_circle_for_triangle(known_points)
    if len(all_points) == 0:
        return circumscribing_circle_for_triangle(known_points)
    i = random.randrange(0, len(all_points))
    p = all_points[i]
    other_p = all_points[:i] + all_points[i+1:]
    ctr, rad = circumscribing_circle_for_polygon(
        other_p,
        known_points.copy()
    )
    if ctr is not None:
        ctr = cast(Coordinate, ctr)
        rad = cast(float, rad)
        if rad >= dist_xyz_meters(p, ctr):
            return ctr, rad
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
    return math.acos(sum([an*bn for an, bn in zip(coord1.xyz, coord2.xyz)])) * EARTH_RADIUS


def do_bounds_overlap(bounds1: Tuple[float, float], bounds2: Tuple[float, float]):
    """
    Test whether two ranges (on the same axis) overlap

    Args:
        bounds1:
            A two-tuple of floats, representing the range of values across the axis
        bounds2:
            Another two-tuple of floats

    Returns:
        True if the ranges overlap, False if not
    """
    return max([bounds1[0], bounds2[0]]) <= min([bounds1[1], bounds2[1]])


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
        do_bounds_overlap(line1_bounds[0], line2_bounds[0]) and
        do_bounds_overlap(line1_bounds[1], line2_bounds[1])
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


def is_point_in_line(point: Coordinate, line: Tuple[Coordinate, Coordinate], **kwargs) -> bool:
    epsilon = kwargs.get('epsilon', 0.00001)

    if not line[0].longitude <= point.longitude <= line[1].longitude:
        # Point does not fall within x bounds of the line
        return False

    if abs(coordinate_vector_cross_product(line[0], line[1], point)) > epsilon:
        # vector cross product differs significantly enough
        return False

    return True


def is_counter_clockwise(bounds: List[Coordinate]) -> bool:
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

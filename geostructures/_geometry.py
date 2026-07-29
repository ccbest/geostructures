"""
Internal module defining geometric functions used by geostructures
"""
import math
import random
from typing import List, Sequence, Tuple, Optional, cast, Set

import numpy as np
from numpy.linalg import norm

from geostructures._const import COORDINATE_TOLERANCE, EARTH_RADIUS_METERS
from geostructures.coordinates import Coordinate
from geostructures.utils.logging import warn_once


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


# Points are shuffled before the incremental construction in minimum_bounding_circle,
# which is what gives Welzl's algorithm its expected-linear runtime. The generator is
# module-private and re-seeded per call so that identical inputs always produce
# identical circles, and so that the caller's own random stream is left alone.
_MBC_SEED = 0

# Circles are grown to fit points that fall outside them, so a point sitting exactly on
# the boundary - every vertex of a regular polygon, for instance - must test as
# contained despite float error.
_MBC_ABS_TOLERANCE_METERS = 1e-6
_MBC_REL_TOLERANCE = 1e-12

# Tolerance for the final enclosure check only, which exists to catch coordinates too
# spread out for a 90 degree cap. That failure misses by hundreds of kilometers, so the
# check can afford to be loose - and needs to be, since it is the one comparison whose
# two sides aren't both measured off the same circle.
_MBC_ENCLOSURE_TOLERANCE_METERS = 1.0


def _circle_contains(
    center: Coordinate,
    radius: float,
    coord: Coordinate,
    tolerance_meters: float = _MBC_ABS_TOLERANCE_METERS
) -> bool:
    """Tests whether a coordinate falls inside a circle, within tolerance."""
    tolerance = max(radius * _MBC_REL_TOLERANCE, tolerance_meters)
    return dist_xyz_meters(center, coord) <= radius + tolerance


def _diameter_circle(coord1: Coordinate, coord2: Coordinate) -> Tuple[Coordinate, float]:
    """
    Produces the circle having two coordinates at opposite ends of its diameter, i.e.
    the midpoint of the two and half the distance between them.

    Args:
        coord1:
            A coordinate

        coord2:
            A second coordinate

    Returns:
        (Coordinate, float) tuple of (center, radius in meters)
    """
    midpoint = [(v1 + v2) / 2 for v1, v2 in zip(coord1.xyz, coord2.xyz)]
    midpoint_norm = norm(midpoint)
    if not midpoint_norm:  # pragma: no cover
        # The two coordinates are antipodal, so every great circle through them is
        # a diameter and the midpoint is undefined
        raise ValueError(
            f'Cannot produce a bounding circle for antipodal coordinates {coord1} '
            f'and {coord2}.'
        )

    center = Coordinate._from_xyz([i / midpoint_norm for i in midpoint])
    return center, dist_xyz_meters(center, coord1)


def _circumcircle(
    coord1: Coordinate,
    coord2: Coordinate,
    coord3: Coordinate
) -> Optional[Tuple[Coordinate, float]]:
    """
    Produces the circle passing through all three coordinates, using the circumcenter
    formula from https://brsr.github.io/2021/05/02/spherical-triangle-centers.html

    Note that this is the circle *through* the three coordinates, which for an obtuse
    triangle is larger than the smallest circle enclosing them.

    Args:
        coord1:
            A coordinate

        coord2:
            A second coordinate

        coord3:
            A third coordinate

    Returns:
        (Coordinate, float) tuple of (circumcenter, radius in meters), or None if the
        coordinates are degenerate (a repeated coordinate, say) and define no circle.
        Note that three coordinates sharing a great circle are not degenerate - that
        great circle is their circumcircle, centered on its pole.
    """
    a, b, c = (np.asarray(coord.xyz, dtype=np.longdouble) for coord in (coord1, coord2, coord3))
    cc_num = np.cross(a, b) + np.cross(b, c) + np.cross(c, a)
    cc_norm = norm(cc_num)
    if not cc_norm:
        return None

    if np.dot(a, np.cross(b, c)) < 0:
        # The coordinates wind clockwise as seen from the circumcenter, so the formula
        # produced its antipode. Negating is cheaper - and correct nearer the poles and
        # antimeridian - than reordering the coordinates by a planar orientation test.
        cc_num = -cc_num

    center = Coordinate._from_xyz([i / cc_norm for i in cc_num])
    # The radius has to be measured the same way _circle_contains measures distance,
    # or the three coordinates defining this circle can test as falling outside it
    return center, dist_xyz_meters(center, coord1)


def _mbc_with_boundary_points(
    points: List[Coordinate],
    coord1: Coordinate,
    coord2: Coordinate
) -> Tuple[Coordinate, float]:
    """
    Supporting function for minimum_bounding_circle(). Produces the smallest circle
    enclosing points that has both coord1 and coord2 on its boundary.

    Candidate circumcircles are tracked separately per side of the great circle running
    through coord1 and coord2, because the enclosing circle must reach the farthest
    outlying point on each side.

    Args:
        points:
            A list of coordinates to enclose

        coord1:
            A coordinate to hold on the boundary

        coord2:
            A second coordinate to hold on the boundary

    Returns:
        (Coordinate, float) tuple of (center, radius in meters)
    """
    circle = _diameter_circle(coord1, coord2)
    # Normal of the great circle through both boundary coordinates; the sign of a
    # coordinate's dot product against it says which side that coordinate falls on
    normal = np.cross(np.asarray(coord1.xyz), np.asarray(coord2.xyz))
    # Each side tracks its best candidate alongside that candidate's own offset from
    # the great circle, which is what ranks the candidates
    left: Optional[Tuple[Coordinate, float]] = None
    right: Optional[Tuple[Coordinate, float]] = None
    left_offset = right_offset = 0.0

    for point in points:
        if _circle_contains(*circle, point):
            continue

        candidate = _circumcircle(coord1, coord2, point)
        if candidate is None:  # pragma: no cover
            # Only a point duplicating one of the two boundary coordinates degenerates
            # here, and such a point is already inside the circle through both of them
            continue

        side = float(np.dot(normal, np.asarray(point.xyz)))
        offset = float(np.dot(normal, np.asarray(candidate[0].xyz)))
        if side > 0 and (left is None or offset > left_offset):
            left, left_offset = candidate, offset
        elif side < 0 and (right is None or offset < right_offset):
            right, right_offset = candidate, offset

    if left is None and right is None:
        return circle
    if left is None:
        return cast(Tuple[Coordinate, float], right)
    if right is None:
        return left
    return left if left[1] <= right[1] else right


def _mbc_with_boundary_point(
    points: List[Coordinate],
    coord: Coordinate
) -> Tuple[Coordinate, float]:
    """
    Supporting function for minimum_bounding_circle(). Produces the smallest circle
    enclosing points that has coord on its boundary.

    Args:
        points:
            A list of coordinates to enclose

        coord:
            A coordinate to hold on the boundary

    Returns:
        (Coordinate, float) tuple of (center, radius in meters)
    """
    circle = (coord, 0.0)
    for idx, point in enumerate(points):
        if _circle_contains(*circle, point):
            continue

        if not circle[1]:
            circle = _diameter_circle(coord, point)
            continue

        circle = _mbc_with_boundary_points(points[:idx + 1], coord, point)

    return circle


def _enclosing_circle_fallback(points: List[Coordinate]) -> Tuple[Coordinate, float]:
    """
    Supporting function for minimum_bounding_circle(). Produces a circle that is
    guaranteed to enclose every coordinate, but is not necessarily the smallest one,
    by measuring out from the mean of the coordinate vectors.

    Args:
        points:
            A list of coordinates to enclose

    Returns:
        (Coordinate, float) tuple of (center, radius in meters)
    """
    mean = [sum(values) / len(points) for values in zip(*(point.xyz for point in points))]
    mean_norm = norm(mean)
    # A zero-length mean means the coordinates cancel out entirely (e.g. antipodal
    # pairs), leaving no meaningful center to measure from
    center = points[0] if not mean_norm else Coordinate._from_xyz([i / mean_norm for i in mean])
    return center, max(dist_xyz_meters(center, point) for point in points)


def minimum_bounding_circle(points: Sequence[Coordinate]) -> Tuple[Coordinate, float]:
    """
    Determines the smallest circle enclosing a set of coordinates, using the incremental
    formulation of Welzl's algorithm. Runs in expected linear time.

    The circle is a spherical cap. Coordinates spanning more than a hemisphere would
    require a cap wider than 90 degrees, which this construction cannot produce; those
    fall back to a circle that still encloses every coordinate but is not the smallest
    one. The returned circle always encloses every coordinate.

    Args:
        points:
            A list of coordinates to enclose. Must not be empty.

    Returns:
        (Coordinate, float) tuple of (center, radius in meters)
    """
    if not points:
        raise ValueError('Cannot produce a bounding circle for zero coordinates.')

    shuffled = list(points)
    random.Random(_MBC_SEED).shuffle(shuffled)

    circle = (shuffled[0], 0.0)
    for idx, point in enumerate(shuffled):
        if _circle_contains(*circle, point):
            continue

        # Any point outside the circle enclosing everything before it has to sit on the
        # boundary of the circle enclosing everything up to and including it
        circle = _mbc_with_boundary_point(shuffled[:idx + 1], point)

    if all(
        _circle_contains(*circle, point, _MBC_ENCLOSURE_TOLERANCE_METERS)
        for point in shuffled
    ):
        return circle

    warn_once(
        'Bounding circle for coordinates spanning more than a hemisphere is not '
        'minimal; this warning will not repeat.'
    )
    return _enclosing_circle_fallback(shuffled)


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
    # For unit vectors, half the length of their difference is sin(theta/2) and half the
    # length of their sum is cos(theta/2), so atan2 recovers the angle between them.
    # Taking acos of their dot product is cheaper but loses nearly all precision for
    # close coordinates - it reported a tenth of a meter between a coordinate and
    # itself - and needs clamping to stay inside acos's domain.
    half_chord = math.hypot(*(a - b for a, b in zip(coord1.xyz, coord2.xyz)))
    half_sum = math.hypot(*(a + b for a, b in zip(coord1.xyz, coord2.xyz)))
    return 2 * math.atan2(half_chord, half_sum) * EARTH_RADIUS_METERS


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
        return coord1, Coordinate(adjusted_lon, coord2.latitude, coord2.z, coord2.m, _bounded=False)
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
        return (x_sort[0], x_sort[1]), (y_sort[0], y_sort[1])

    def within_bounds(value: float, bounds: Tuple[float, float]) -> bool:
        """Tests a value against a min/max pair, tolerant of coordinate-level error"""
        return bounds[0] - COORDINATE_TOLERANCE <= value <= bounds[1] + COORDINATE_TOLERANCE

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
    x_intersection = det(d, xdiff) / div
    y_intersection = det(d, ydiff) / div

    if (
        within_bounds(x_intersection, line1_bounds[0]) and
        within_bounds(x_intersection, line2_bounds[0]) and
        within_bounds(y_intersection, line1_bounds[1]) and
        within_bounds(y_intersection, line2_bounds[1])
    ):
        return (
            Coordinate(x_intersection, y_intersection),

            # If intersecting point is a line endpoint, is a boundary intersection
            any(
                math.isclose(x_intersection, endpoint[0], abs_tol=COORDINATE_TOLERANCE)
                and math.isclose(y_intersection, endpoint[1], abs_tol=COORDINATE_TOLERANCE)
                for endpoint in (*line1_flt, *line2_flt)
            )
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


def polygon_centroid(rings: Sequence[Sequence[Coordinate]]) -> Tuple[float, float]:
    """
    Computes the area-weighted centroid of one or more closed rings by decomposing
    them into triangles using vertex pairs around the origin.

    Rings are concatenated before decomposition. This is safe because the edges
    joining one ring to the next are traversed in both directions, so their
    contributions cancel out.

    Args:
        rings:
            Lists of self-closing Coordinates, in order. Z and M values, if
            present, are ignored.

    Returns:
        A 2-tuple of the centroid's longitude and latitude
    """
    # Slice to lon/lat; to_float() may also carry Z/M values
    poly1 = np.array([coord.to_float()[:2] for ring in rings for coord in ring])
    poly2 = np.roll(poly1, -1, axis=0)

    # Compute signed area manually since np.cross is deprecated for 2D inputs
    signed_areas = 0.5 * (poly1[:, 0] * poly2[:, 1] - poly1[:, 1] * poly2[:, 0])

    if not signed_areas.sum():
        # Degenerate (zero-area) rings, e.g. collinear or duplicate coordinates, have
        # no area to weight by. Fall back to the mean of the vertices, dropping each
        # ring's closing coordinate so the first vertex isn't counted twice.
        lon, lat = np.mean(
            np.array([coord.to_float()[:2] for ring in rings for coord in ring[:-1]]),
            axis=0
        )
        return float(lon), float(lat)

    # Return average of triangle centroids, weighted by area
    lon, lat = np.average((poly1 + poly2) / 3, axis=0, weights=signed_areas)
    return float(lon), float(lat)


def convert_trig_angle(bearing: float) -> float:
    """
    Converts angles between bearing (angle of 0 corresponds to the point (1, 0) and increases
    in the clockwise direction) and trigonometric (angle of 0 corresponds to the point (1,0)
    and increases in the counterclockwise direction) definitions.

    This function is its own inverse, meaning calling it twice will produce the same input

    Args:
        bearing: (float)
            An angle of bearing

    Returns:
        float
    """
    return (90 - bearing) % 360

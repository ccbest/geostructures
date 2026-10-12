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


# Gauss-Legendre quadrature over [-1, 1], used to evaluate Owen's T function
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(24)

# Polygon edges are subdivided prior to projection, so that distortion from the
# projection is negligible. Subdivisions are no longer than _MAX_EDGE_DEGREES, and short
# enough that their deviation from the true edge is within _EDGE_TOLERANCE standard
# deviations, but no edge is split into more than _MAX_EDGE_SUBDIVISIONS pieces.
_MAX_EDGE_DEGREES = 0.5
_EDGE_TOLERANCE = 1e-3
_MAX_EDGE_SUBDIVISIONS = 1000


def _std_normal_cdf(x: np.ndarray) -> np.ndarray:
    """The standard normal CDF, Phi(x), computed with erfc to preserve tail precision"""
    return np.frompyfunc(math.erfc, 1, 1)(-x / math.sqrt(2)).astype(float) / 2


def _owens_t(h: np.ndarray, a: np.ndarray) -> np.ndarray:
    """
    Owen's T function, T(h, a), for 0 <= a <= 1. Over that range the integrand is
    smooth, so Gauss-Legendre quadrature is highly accurate.

    Args:
        h:
            An array of h values

        a:
            An array of a values, each between 0 and 1

    Returns:
        An array of T(h, a)
    """
    x = a[:, None] * (_GL_NODES[None, :] + 1) / 2  # Map nodes onto [0, a]
    integrand = np.exp(-0.5 * h[:, None] ** 2 * (1 + x ** 2)) / (1 + x ** 2)
    return a / 2 * (integrand @ _GL_WEIGHTS) / (2 * math.pi)


def _std_normal_right_triangle_probability(c: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    The probability mass of the standard bivariate normal (centered on the origin)
    over the right triangle with vertices at:
        * the origin
        * a point F, distance c from the origin
        * a point distance s from F, perpendicular to the origin-F segment

    This equals atan(s / c) / 2pi - T(c, s / c), where T is Owen's T function. For
    s > c, the identity T(h, a) + T(ah, 1/a) = (Phi(h) + Phi(ah)) / 2 - Phi(h)Phi(ah)
    keeps the T function argument within [0, 1].

    Args:
        c:
            An array of non-negative distances from the origin to F

        s:
            An array of non-negative distances from F to the third vertex

    Returns:
        An array of probabilities
    """
    short = s <= c
    num, den = np.where(short, s, c), np.where(short, c, s)
    ratio = np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)
    t = _owens_t(np.where(short, c, s), ratio)

    cdf_c, cdf_s = _std_normal_cdf(c), _std_normal_cdf(s)
    # (Phi(c) + Phi(s)) / 2 - Phi(c)Phi(s), rearranged to use upper tails for precision
    reflected = (cdf_c * _std_normal_cdf(-s) + cdf_s * _std_normal_cdf(-c)) / 2 - t

    return np.arctan2(s, c) / (2 * math.pi) - np.where(short, t, reflected)


def _project_to_whitened_plane(
    lonlats: np.ndarray,
    center: Coordinate,
    sigma_major: float,
    sigma_minor: float,
    rotation: float,
) -> np.ndarray:
    """
    Projects lon/lat points onto an azimuthal equidistant plane about a center
    point (preserving distance and bearing from the center, as GeoEllipse does),
    then scales and rotates the plane such that a Gaussian with the given standard
    deviations becomes a standard bivariate normal.

    Args:
        lonlats:
            An (n, 2) array of longitude/latitude pairs, in degrees

        center:
            The center of the Gaussian

        sigma_major:
            The standard deviation along the major axis, in meters

        sigma_minor:
            The standard deviation along the minor axis, in meters

        rotation:
            The major axis's degree offset from North (expressed as East of North)

    Returns:
        An (n, 2) array of projected points
    """
    lon, lat = np.radians(lonlats[:, 0]), np.radians(lonlats[:, 1])
    lon0, lat0 = math.radians(center.longitude), math.radians(center.latitude)
    d_lon = lon - lon0

    hav = np.clip(
        np.sin((lat - lat0) / 2) ** 2 + math.cos(lat0) * np.cos(lat) * np.sin(d_lon / 2) ** 2,
        0.,
        1.
    )
    dist = EARTH_RADIUS * 2 * np.arctan2(np.sqrt(hav), np.sqrt(1 - hav))
    bearing = np.arctan2(
        np.sin(d_lon) * np.cos(lat),
        math.cos(lat0) * np.sin(lat) - math.sin(lat0) * np.cos(lat) * np.cos(d_lon)
    )
    east, north = dist * np.sin(bearing), dist * np.cos(bearing)

    rot = math.radians(rotation)
    return np.stack(
        [
            (east * math.sin(rot) + north * math.cos(rot)) / sigma_major,
            (east * math.cos(rot) - north * math.sin(rot)) / sigma_minor,
        ],
        axis=1
    )


def gaussian_probability_in_polygons(
    rings: List[List[Coordinate]],
    center: Coordinate,
    sigma_major: float,
    sigma_minor: float,
    rotation: float,
) -> List[float]:
    """
    Integrates a 2-D Gaussian over each of a list of polygons.

    The Gaussian is defined on an azimuthal equidistant projection about its center,
    with the major axis oriented `rotation` degrees East of North. Polygon edges are
    treated as straight lines in lon/lat space.

    Each polygon is projected into a space where the Gaussian is a standard bivariate
    normal, and decomposed into triangles fanning out from the origin. The mass of
    each triangle is computed exactly (to quadrature precision) via Owen's T function.

    Args:
        rings:
            A list of linear rings, each a list of Coordinates. Rings may be open
            or closed.

        center:
            The center of the Gaussian

        sigma_major:
            The standard deviation along the major axis, in meters

        sigma_minor:
            The standard deviation along the minor axis, in meters

        rotation:
            The major axis's degree offset from North (expressed as East of North)

    Returns:
        A list of probabilities, one per ring
    """
    if not rings:
        return []

    verts = [np.array([coord.to_float() for coord in ring]) for ring in rings]
    starts = np.concatenate(verts)
    ring_ids = np.repeat(np.arange(len(rings)), [len(x) for x in verts])
    deltas = np.concatenate([np.roll(x, -1, axis=0) for x in verts]) - starts
    # Take the short way around if an edge crosses the antimeridian
    deltas[:, 0] = (deltas[:, 0] + 180) % 360 - 180

    # Subdivide long edges, which would otherwise be distorted by the projection
    max_step = min(
        _MAX_EDGE_DEGREES,
        math.degrees(math.sqrt(8 * _EDGE_TOLERANCE * sigma_minor / EARTH_RADIUS))
    )
    n_sub = np.clip(
        np.ceil(np.abs(deltas).max(axis=1) / max_step), 1, _MAX_EDGE_SUBDIVISIONS
    ).astype(int)
    edge_idx = np.repeat(np.arange(len(starts)), n_sub)
    frac = (np.arange(n_sub.sum()) - np.repeat(np.cumsum(n_sub) - n_sub, n_sub)) / n_sub[edge_idx]
    seg_starts = starts[edge_idx] + frac[:, None] * deltas[edge_idx]
    seg_ends = seg_starts + deltas[edge_idx] / n_sub[edge_idx][:, None]

    point_a = _project_to_whitened_plane(seg_starts, center, sigma_major, sigma_minor, rotation)
    point_b = _project_to_whitened_plane(seg_ends, center, sigma_major, sigma_minor, rotation)

    # Each segment AB forms a triangle with the origin. Splitting it at F, the foot of
    # the perpendicular from the origin to line AB, gives two right triangles.
    diff = point_b - point_a
    length = np.hypot(diff[:, 0], diff[:, 1])
    unit = np.divide(diff, length[:, None], out=np.zeros_like(diff), where=length[:, None] > 0)
    s_a = (point_a * unit).sum(axis=1)  # signed distances from F along AB
    s_b = (point_b * unit).sum(axis=1)
    cross = point_a[:, 0] * unit[:, 1] - point_a[:, 1] * unit[:, 0]
    c = np.abs(cross)

    def _signed_mass(s):
        return np.sign(s) * _std_normal_right_triangle_probability(c, np.abs(s))

    # Signs cancel out the mass outside the polygon once summed around the ring
    triangle_mass = np.sign(cross) * (_signed_mass(s_b) - _signed_mass(s_a))
    totals = np.bincount(ring_ids[edge_idx], weights=triangle_mass, minlength=len(rings))

    return np.clip(np.abs(totals), 0., 1.).tolist()


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

""" Geometric calculations for Coordinates and Geostructures """

__all__ = [
    'bearing_degrees', 'haversine_distance_meters', 'inverse_haversine_degrees',
    'inverse_haversine_radians', 'rotate_coordinates', 'find_line_intersection'
]

import math
from typing import List, Optional, Tuple

import numpy as np

from geostructures.coordinates import Coordinate
from geostructures.utils.functions import round_half_up


_EARTH_RADIUS = 6_371_000  # meters - WGS84


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


def test_intersection(
    vertices_a: List[Tuple[Coordinate, Coordinate]],
    vertices_b: List[Tuple[Coordinate, Coordinate]],
) -> bool:
    """
    Tests whether two sets of vertices ever intersect. Uses the sweep line algorithm
    to minimize the number of intersections calculated.

    Args:
        vertices_a:
            The list of vertices from the first group/shape

        vertices_b:
            The list of vertices from the second group/shape

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

    def _create_events(vertices, group):
        """Creates 2x events per vertex from a list of vertices. Ensures the lesser x
        value corresponds to the start event."""
        _events = []
        for vertex in vertices:
            if vertex[0].latitude > vertex[1].latitude:
                vertex = (vertex[1], vertex[0])

            _events += [
                _Event(vertex[0].latitude, vertex[0].latitude <= vertex[1].latitude, vertex, group),
                _Event(vertex[1].latitude, vertex[1].latitude <= vertex[0].latitude, vertex, group)
            ]
        return _events

    events = _create_events(vertices_a, 'a')
    events += _create_events(vertices_b, 'b')

    events.sort()
    active_events = set()
    for event in events:
        if not event.is_start:
            active_events.remove(event)
            continue

        # All vertices belong to same group
        if len(set(x.group for x in active_events)) <= 1:
            active_events.add(event)
            continue

        for active_event in active_events:
            if event.group == active_event.group:
                continue

            intersection = find_line_intersection(active_event.segment, event.segment)
            if intersection and not intersection[1]:
                return True

        active_events.add(event)

    return False


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
    angle = np.deg2rad(degrees)
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    o = np.atleast_2d(origin.to_float())
    p = np.atleast_2d([x.to_float() for x in coords])

    return [
        Coordinate(*[round_half_up(x, precision) for x in coord])
        for coord in (R @ (p.T - o.T) + o.T).T
    ]


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

    line1_flt = (line1[0].to_float(), line1[1].to_float())
    if line1_flt[1][0] < line1_flt[0][0]:  # Flip order such that lower x value is first
        line1_flt = (line1_flt[1], line1_flt[0])

    line2_flt = (line2[0].to_float(), line2[1].to_float())
    if line2_flt[1][0] < line2_flt[0][0]:
        line2_flt = (line2_flt[1], line2_flt[0])

    line1_y_bounds = (
        min([line1_flt[0][1], line1_flt[1][1]]),
        max([line1_flt[0][1], line1_flt[1][1]])
    )
    line2_y_bounds = (
        min([line2_flt[0][1], line2_flt[1][1]]),
        max([line2_flt[0][1], line2_flt[1][1]])
    )

    if not (
            max([line1_flt[0][0], line2_flt[0][0]]) <= min([line1_flt[1][0], line2_flt[1][0]]) and
            max(
                [line1_y_bounds[0], line2_y_bounds[0]]
            ) <= min(
                [line1_y_bounds[1], line2_y_bounds[1]]
            )
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

    # Check if any of the x values are exactly the same - could be boundary intersection
    if (x_intersection, y_intersection) in (*line1_flt, *line2_flt):
        # Intersection exactly on one of the coordinates - boundary intersection
        return Coordinate(x_intersection, y_intersection), True

    if (
            max(
                [line1_flt[0][0], line2_flt[0][0]]
            ) <= x_intersection <= min(
                [line1_flt[1][0], line2_flt[1][0]]
            )
            and max(
                [line1_y_bounds[0], line2_y_bounds[0]]
            ) <= y_intersection <= min(
                [line1_y_bounds[1], line2_y_bounds[1]]
            )
    ):
        return Coordinate(x_intersection, y_intersection), False

    return None

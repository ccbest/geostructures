""" Geometric calculations for Coordinates and Geostructures """

__all__ = [
    'bearing_degrees', 'haversine_distance_meters', 'inverse_haversine_degrees',
    'inverse_haversine_radians', 'rotate_coordinates'
]

import math
from typing import List, Tuple

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
    d_lon = float(coord2.longitude) - float(coord1.longitude)
    x_val = math.cos(math.radians(float(coord2.latitude))) * math.sin(
        math.radians(d_lon)
    )
    y_val = math.cos(math.radians(float(coord1.latitude))) * math.sin(
        math.radians(float(coord2.latitude))
    ) - math.sin(math.radians(float(coord1.latitude))) * math.cos(
        math.radians(float(coord2.latitude))
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

    d_lat = float(lat2 - lat1)
    d_long = float(lon2 - lon1)
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

    x0 = float(start.longitude) * math.pi / 180
    y0 = float(start.latitude) * math.pi / 180

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
        degrees: float
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

    precision = [min([x.latitude.precision, x.longitude.precision]) for x in coords]
    return [
        Coordinate(*(round_half_up(_x, y) for _x in x))
        for x, y in zip((R @ (p.T - o.T) + o.T).T, precision)
    ]


def find_line_intersection(
        line1: Tuple[Coordinate, Coordinate],
        line2: Tuple[Coordinate, Coordinate]
):
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    line1 = (line1[0].to_float(), line1[1].to_float())
    if line1[1][0] < line1[0][0]:  # Flip order such that lower x value is first
        line1 = (line1[1], line1[0])

    line2 = (line2[0].to_float(), line2[1].to_float())
    if line2[1][0] < line2[0][0]:
        line2 = (line2[1], line2[0])

    line1_y_bounds = min([line1[0][1], line1[1][1]]), max([line1[0][1], line1[1][1]])
    line2_y_bounds = min([line2[0][1], line2[1][1]]), max([line2[0][1], line2[1][1]])

    if not (
            max([line1[0][0], line2[0][0]]) <= min([line1[1][0], line2[1][0]]) and
            max([line1_y_bounds[0], line2_y_bounds[0]]) <= min([line1_y_bounds[1], line2_y_bounds[1]])
    ):
        # line bounds do not overlap
        return

    line1_diff = (line1[0][0] - line1[1][0], line1[0][1] - line1[1][1])
    xdiff = (line1_diff[0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    div = det(xdiff, (line1_diff[1], line2[0][1] - line2[1][1]))
    if div == 0:
        # lines are parallel
        return

    d = (det(*line1), det(*line2))
    x_intersection = det(d, xdiff) / div
    y_intersection = det(d, ydiff) / div

    if (
            max([line1[0][0], line2[0][0]]) <= x_intersection <= min([line1[1][0], line2[1][0]])
            and max(
                [line1_y_bounds[0], line2_y_bounds[0]]
            ) <= y_intersection <= min(
                [line1_y_bounds[1], line2_y_bounds[1]]
            )
    ):
        return x_intersection, y_intersection

    return


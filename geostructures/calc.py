""" Geometric calculations for Coordinates and Geostructures """

__all__ = [
    'bearing_degrees', 'haversine_distance_meters', 'inverse_haversine_degrees',
    'inverse_haversine_radians', 'rotate_coordinates'
]

import math
from typing import List, Tuple, cast

import numpy as np

from geostructures._const import EARTH_RADIUS
from geostructures._geometry import ensure_edge_bounds
from geostructures.coordinates import Coordinate
from geostructures.utils.functions import round_half_up


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
    coord1, coord2 = ensure_edge_bounds(coord1, coord2)

    lon1, lat1 = math.radians(coord1.longitude), math.radians(coord1.latitude)
    lon2, lat2 = math.radians(coord2.longitude), math.radians(coord2.latitude)

    d_lat, d_long = lat2 - lat1, lon2 - lon1
    var1 = (math.sin(d_lat / 2) ** 2) + math.cos(lat1) * math.cos(lat2) * (
        math.sin(d_long / 2) ** 2
    )
    return EARTH_RADIUS * 2 * math.atan2(math.sqrt(var1), math.sqrt(1 - var1))


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

    Coordinate Z values will be preserved, however M values will not.

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
    _rad = distance_meters / EARTH_RADIUS

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

    return Coordinate(final_lon, final_lat, z=start.z)


def rotate_coordinates(
        coords: List[Coordinate],
        origin: Coordinate,
        degrees: float,
) -> List[Coordinate]:
    """
    Rotate a set of coordinates around an origin. Returned Coordinate precision will be
    determined by the respective Coordinate's least significant precision.

    Coordinate Z values will be preserved, however M values will not.

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
    coords = [ensure_edge_bounds(origin, x)[1] for x in coords]
    angle = np.deg2rad(degrees)
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    z = (x.z for x in coords)
    o = np.atleast_2d(origin.to_float()[:2])
    p = np.atleast_2d([x.to_float()[:2] for x in coords])
    return [
        Coordinate(*cast(Tuple[float, float], coord), zval)
        for coord, zval in zip((R @ (p.T - o.T) + o.T).T, z)
    ]

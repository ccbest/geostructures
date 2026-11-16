"""
Bounding boxes for curved shapes (GeoCircle, GeoEllipse, GeoRing).

A box is (min_lon, min_lat, max_lon, max_lat). Longitudes are computed
relative to the shape's center, then each edge is wrapped to [-180, 180),
so a shape crossing the antimeridian has its west edge greater than its east.
"""

__all__ = ['circle_bounds', 'sampled_bounds']

import math
from typing import Tuple

import numpy as np

from geostructures._const import EARTH_RADIUS_METERS
from geostructures.coordinates import Coordinate
from geostructures.geodesic import (
    _active_is_haversine, _haversine_destination_array,
    destination_point, distance_meters,
)

# Samples taken around a full turn when an outline has no closed form
_SAMPLES = 256

# Golden-section search for the widest bearing of a circle on the ellipsoid
_GOLDEN = (math.sqrt(5) - 1) / 2
_SEARCH_HALF_WIDTH = 15.
_SEARCH_ITERATIONS = 40


def _relative_lon(lon: float, center_lon: float) -> float:
    """Longitude relative to center_lon, in [-180, 180)"""
    return (lon - center_lon + 180.) % 360. - 180.


def _wrap_lon(lon: float) -> float:
    """Wraps a longitude to [-180, 180), as Coordinate does"""
    if -180. <= lon < 180.:
        return lon
    return Coordinate(lon, 0.).longitude


def _widest_offset(center: Coordinate, radius: float, guess: float) -> float:
    """
    The largest eastward longitude offset (degrees) reached by a circle under
    the active geodesic algorithm, found by golden-section search over the
    bearing around `guess` (degrees).
    """
    def offset(bearing: float) -> float:
        return _relative_lon(
            destination_point(center, bearing, radius).longitude, center.longitude
        )

    low = max(guess - _SEARCH_HALF_WIDTH, 0.)
    high = min(guess + _SEARCH_HALF_WIDTH, 180.)
    left = high - _GOLDEN * (high - low)
    right = low + _GOLDEN * (high - low)
    f_left, f_right = offset(left), offset(right)
    for _ in range(_SEARCH_ITERATIONS):
        if f_left < f_right:
            low, left, f_left = left, right, f_right
            right = low + _GOLDEN * (high - low)
            f_right = offset(right)
        else:
            high, right, f_right = right, left, f_left
            left = high - _GOLDEN * (high - low)
            f_left = offset(left)

    return max(f_left, f_right)


def circle_bounds(center: Coordinate, radius: float) -> Tuple[float, float, float, float]:
    """
    The bounding box of a geodesic circle.

    Latitudes come from the north and south points. Under Haversine the
    longitude half-width is asin(sin(d) / cos(lat)) for angular radius d;
    under other algorithms it is found by searching the bearing near the
    Haversine answer. A circle reaching a pole spans every longitude.

    Args:
        center:
            The circle's center

        radius:
            The circle's radius, in meters

    Returns:
        (min_lon, min_lat, max_lon, max_lat)
    """
    ang_dist = radius / EARTH_RADIUS_METERS
    lat_rad = math.radians(center.latitude)
    haversine = _active_is_haversine()

    if haversine:
        # Due north and due south of center, without building Coordinates
        min_lat = math.degrees(lat_rad - ang_dist)
        max_lat = math.degrees(lat_rad + ang_dist)
    else:
        min_lat = destination_point(center, 180., radius).latitude
        max_lat = destination_point(center, 0., radius).latitude

    # Cheap test first; a degree of latitude on the ellipsoid is at most ~1% shorter
    if abs(center.latitude) + 1.01 * math.degrees(ang_dist) >= 90.:
        reaches_north = distance_meters(center, Coordinate(0., 90.)) <= radius
        reaches_south = distance_meters(center, Coordinate(0., -90.)) <= radius
        if reaches_north or reaches_south:
            return (
                -180.,
                -90. if reaches_south else min_lat,
                180.,
                90. if reaches_north else max_lat,
            )

    if haversine:
        half_width = math.degrees(
            math.asin(min(math.sin(ang_dist) / math.cos(lat_rad), 1.))
        )
    else:
        # Bearing of the widest point on a sphere: cos(bearing) = tan(lat) * tan(d)
        guess = math.degrees(
            math.acos(max(min(math.tan(lat_rad) * math.tan(ang_dist), 1.), -1.))
        )
        # A circle is symmetric about its center's meridian, so west mirrors east
        half_width = _widest_offset(center, radius, guess)

    return (
        _wrap_lon(center.longitude - half_width),
        min_lat,
        _wrap_lon(center.longitude + half_width),
        max_lat,
    )


def sampled_bounds(
    center: Coordinate,
    bearings: np.ndarray,
    distances: np.ndarray,
    largest_radius: float,
) -> Tuple[float, float, float, float]:
    """
    The bounding box of points placed from `center` at `bearings` and
    `distances`, padded on each side by the most a curved outline sampled
    every 360 / 256 degrees can bulge between two samples.

    Args:
        center:
            The shape's center

        bearings:
            Bearings of the sample points, in degrees

        distances:
            Distances of the sample points from center, in meters

        largest_radius:
            The shape's largest radius, in meters

    Returns:
        (min_lon, min_lat, max_lon, max_lat)
    """
    if _active_is_haversine():
        lons, lats = _haversine_destination_array(center, bearings, distances)
        rel_lons = lons - center.longitude
    else:
        points = [
            destination_point(center, bearing, distance)
            for bearing, distance in zip(bearings.tolist(), distances.tolist())
        ]
        lats = np.array([point.latitude for point in points])
        rel_lons = (
            np.array([point.longitude for point in points]) - center.longitude + 180.
        ) % 360. - 180.

    # Bulge between two samples, and the arc between them, as angles in degrees
    pad = math.degrees(
        largest_radius * (1. - math.cos(math.pi / _SAMPLES)) / EARTH_RADIUS_METERS
    )
    step = math.degrees(largest_radius * 2. * math.pi / _SAMPLES / EARTH_RADIUS_METERS)

    def lon_pad(index: int) -> float:
        # Degrees of longitude shrink poleward; use the most poleward latitude in reach
        lat = min(abs(float(lats[index])) + step, 89.999999)
        return pad / math.cos(math.radians(lat))

    west, east = int(np.argmin(rel_lons)), int(np.argmax(rel_lons))
    return (
        _wrap_lon(center.longitude + float(rel_lons[west]) - lon_pad(west)),
        max(float(lats.min()) - pad, -90.),
        _wrap_lon(center.longitude + float(rel_lons[east]) + lon_pad(east)),
        min(float(lats.max()) + pad, 90.),
    )

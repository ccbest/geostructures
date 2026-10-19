"""
Vectorized geometry kernels: numpy versions of the per-coordinate tests in
geostructures.coordinates, geodesic and structures, operating on arrays of
longitudes and latitudes. Each mirrors its scalar counterpart operation for
operation, so results agree to floating point noise.
"""

__all__ = [
    'bound_coordinates', 'coordinate_equals_mask', 'haversine_meters', 'isclose',
    'optional_equals_mask', 'points_in_ring', 'shape_contains_mask',
]

import math
from typing import Optional, Sequence

import numpy as np

from geostructures import geodesic
from geostructures._base import BaseShape
from geostructures._const import COORDINATE_TOLERANCE, EARTH_RADIUS_METERS
from geostructures._geometry import ensure_edge_bounds
from geostructures.coordinates import Coordinate
from geostructures.multistructures import MultiGeoPolygon
from geostructures.structures import GeoBox, GeoCircle, GeoPolygon, PolygonBase


def bound_coordinates(lon: np.ndarray, lat: np.ndarray) -> None:
    """
    Wraps longitudes and latitudes into range, in place, using the same steps
    (and therefore the same floating point results) as Coordinate.__init__.
    """
    out = np.flatnonzero(~((-90 <= lat) & (lat <= 90)))
    while out.size:
        # Crosses one of the poles
        _lat, _lon = lat[out], lon[out]
        lat[out] = np.where(_lat > 90, 90 - (_lat - 90), -90 - (_lat + 90))
        lon[out] = np.where(_lon < 0, _lon + 180, _lon - 180)
        out = out[~((-90 <= lat[out]) & (lat[out] <= 90))]

    out = np.flatnonzero(~((-180 <= lon) & (lon <= 180)))
    while out.size:
        # Crosses the antimeridian
        _lon = lon[out]
        lon[out] = np.where(_lon > 180, _lon - 360, _lon + 360)
        out = out[~((-180 <= lon[out]) & (lon[out] <= 180))]

    # Longitudes are bounded to [-180, 180)
    lon[lon == 180] = -180


def coordinate_equals_mask(
    lon: np.ndarray,
    lat: np.ndarray,
    z: Optional[np.ndarray],
    m: Optional[np.ndarray],
    coord: Coordinate
) -> np.ndarray:
    """Vectorized Coordinate.__eq__ against a single coordinate"""
    return (
        (lon == coord.longitude) & (lat == coord.latitude) &
        optional_equals_mask(z, coord.z, lon.shape[0]) &
        optional_equals_mask(m, coord.m, lon.shape[0])
    )


def haversine_meters(lon: np.ndarray, lat: np.ndarray, center: Coordinate) -> np.ndarray:
    """Vectorized geodesic.haversine_distance(coord, center), operation for operation"""
    lon1, lat1 = np.radians(lon), np.radians(lat)
    lon2, lat2 = math.radians(center.longitude), math.radians(center.latitude)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * math.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return EARTH_RADIUS_METERS * c


def isclose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise math.isclose(a, b, abs_tol=COORDINATE_TOLERANCE) with its default rel_tol"""
    tol = np.maximum(1e-9 * np.maximum(np.abs(a), np.abs(b)), COORDINATE_TOLERANCE)
    return (a == b) | (np.abs(a - b) <= tol)


def optional_equals_mask(arr: Optional[np.ndarray], value: Optional[float], size: int) -> np.ndarray:
    """Equality against an optional Z/M value, where NaN (or no array at all) stands for None"""
    if value is None:
        return np.ones(size, dtype=bool) if arr is None else np.isnan(arr)

    if arr is None:
        return np.zeros(size, dtype=bool)

    return arr == value


def points_in_ring(lon: np.ndarray, lat: np.ndarray, ring: Sequence[Coordinate]) -> np.ndarray:
    """
    Vectorized GeoPolygon._point_in_polygon (with include_boundary=False): an
    eastward ray cast from each point, with points on the boundary excluded.
    Loops over the ring's edges and is vectorized over the points.
    """
    odd_crossings = np.zeros(lon.shape, dtype=bool)
    on_boundary = np.zeros(lon.shape, dtype=bool)

    for edge in zip(ring, [*ring[1:], ring[0]]):
        # Adjust edges if they cross the antimeridian
        start, end = ensure_edge_bounds(edge[0], edge[1])
        lat_a, lat_b = start.latitude, end.latitude
        lon_a, lon_b = start.longitude, end.longitude

        # Shift the test points into the edge's longitudinal frame, in case
        # the edge was unbounded across the antimeridian
        mid_lon = (lon_a + lon_b) / 2
        p_lon = np.where(lon - mid_lon > 180, lon - 360, np.where(mid_lon - lon > 180, lon + 360, lon))

        if lat_a == lat_b:
            # Horizontal edge; the ray can only lie along it, never cross it
            on_boundary |= (lat == lat_a) & (min(lon_a, lon_b) <= p_lon) & (p_lon <= max(lon_a, lon_b))
            continue

        # Longitude at which the edge's line crosses each point's latitude
        lon_cross = lon_a + (lat - lat_a) * (lon_b - lon_a) / (lat_b - lat_a)

        on_boundary |= (
            (min(lat_a, lat_b) <= lat) & (lat <= max(lat_a, lat_b)) & isclose(lon_cross, p_lon)
        )

        # Half-open rule, so an edge endpoint exactly at a point's latitude is
        # attributed to only one of its two adjacent edges
        odd_crossings ^= ((lat_a > lat) != (lat_b > lat)) & (lon_cross > p_lon)

    return odd_crossings & ~on_boundary


def shape_contains_mask(shape: BaseShape, lon: np.ndarray, lat: np.ndarray) -> Optional[np.ndarray]:
    """
    Vectorized shape.contains_coordinate() over arrays of coordinates (2D;
    Z/M never affect polygon containment). Returns None when the shape, or one
    of its holes, has no vectorized implementation.
    """
    if isinstance(shape, GeoBox):
        mask = (
            (shape.nw_bound.longitude <= lon) & (lon <= shape.se_bound.longitude) &
            (shape.se_bound.latitude <= lat) & (lat <= shape.nw_bound.latitude)
        )

    elif isinstance(shape, GeoCircle):
        # Only Haversine is vectorized; other geodesic algorithms fall back
        if geodesic._active_distance is not geodesic.haversine_distance:
            return None
        mask = haversine_meters(lon, lat, shape.center) <= shape.radius

    elif isinstance(shape, GeoPolygon):
        # Bounding rectangle first, then the ray cast for whatever falls inside it
        min_lon, min_lat, max_lon, max_lat = shape.bounds
        mask = (min_lon <= lon) & (lon <= max_lon) & (min_lat <= lat) & (lat <= max_lat)
        idx = np.flatnonzero(mask)
        mask[idx] = points_in_ring(lon[idx], lat[idx], shape.outline)

    elif isinstance(shape, MultiGeoPolygon):
        mask = np.zeros(lon.shape, dtype=bool)
        for subshape in shape.geoshapes:
            submask = shape_contains_mask(subshape, lon, lat)
            if submask is None:
                return None
            mask |= submask
        return mask

    else:
        return None

    if isinstance(shape, PolygonBase):
        for hole in shape.holes:
            idx = np.flatnonzero(mask)
            in_hole = shape_contains_mask(hole, lon[idx], lat[idx])
            if in_hole is None:
                return None
            mask[idx] = ~in_hole

    return mask

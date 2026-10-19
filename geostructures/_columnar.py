"""
Columnar (array) views of shape collections.

Collections derive these arrays from their shapes and use them to answer
whole-collection questions - time and spatial filtering - with numpy rather
than one shape at a time. They are purely an acceleration structure: every
result computed here must match what the per-shape methods would return, and
the tests compare the two directly.
"""

__all__ = ['ShapeColumns', 'shape_envelope', 'points_in_shape']

from datetime import datetime, timedelta, timezone
from functools import cached_property
import math
from typing import Callable, Dict, Optional, Sequence, Tuple, Type

import numpy as np

from geostructures._base import BaseShape
from geostructures._const import COORDINATE_TOLERANCE, EARTH_RADIUS_METERS
from geostructures._geometry import ensure_edge_bounds
from geostructures.coordinates import Coordinate
from geostructures.multistructures import MultiGeoLineString, MultiGeoPoint, MultiGeoPolygon
from geostructures.structures import (
    GeoBox, GeoCircle, GeoEllipse, GeoLineString, GeoPoint, GeoPolygon, GeoRing
)
from geostructures.time import TimeInterval


_ENVELOPE = Tuple[float, float, float, float]

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_MICROSECOND = timedelta(microseconds=1)

# Envelopes are padded well beyond the tolerance the exact predicates allow
# when deciding whether edges touch
_ENVELOPE_PAD = 10 * COORDINATE_TOLERANCE

# Curved shapes are enveloped by a spherical cap. Ellipsoidal geodesic
# algorithms (vincenty, karney) can reach ~0.6% further in latitude than the
# spherical earth, so the cap radius is inflated to cover them
_CAP_RADIUS_FACTOR = 1.01

# Mirrors math.isclose's default, used by GeoPolygon._point_in_polygon
_ISCLOSE_REL_TOL = 1e-9


def _to_micros(dt: datetime) -> int:
    """Exact microseconds since the epoch; TimeInterval datetimes are always tz-aware"""
    return (dt - _EPOCH) // _MICROSECOND


def _union(*envelopes: _ENVELOPE) -> _ENVELOPE:
    min_lons, min_lats, max_lons, max_lats = zip(*envelopes)
    return min(min_lons), min(min_lats), max(max_lons), max(max_lats)


def _vertex_envelope(coords: Sequence[Coordinate]) -> _ENVELOPE:
    """
    The envelope of a chain of straight edges. Edges that cross the antimeridian
    (see ensure_edge_bounds) are not bounded in longitude, so a chain containing
    one spans every longitude.
    """
    lons = [c.longitude for c in coords]
    lats = [c.latitude for c in coords]
    if any(abs(a - b) > 180 for a, b in zip(lons, lons[1:])):
        return -math.inf, min(lats), math.inf, max(lats)
    return min(lons), min(lats), max(lons), max(lats)


def _cap_envelope(center: Coordinate, radius: float) -> _ENVELOPE:
    """The envelope of every point within radius meters of center"""
    delta = math.degrees(radius * _CAP_RADIUS_FACTOR / EARTH_RADIUS_METERS)
    lat = center.latitude
    min_lat, max_lat = lat - delta, lat + delta
    if min_lat <= -90 or max_lat >= 90:
        # Covers a pole, and therefore every longitude
        return -math.inf, max(min_lat, -90.), math.inf, min(max_lat, 90.)

    # The widest point of a spherical cap isn't due east/west of its center
    half_width = math.degrees(math.asin(min(
        1., math.sin(math.radians(delta)) / math.cos(math.radians(lat))
    )))
    min_lon, max_lon = center.longitude - half_width, center.longitude + half_width
    if min_lon < -180 or max_lon > 180:
        return -math.inf, min_lat, math.inf, max_lat
    return min_lon, min_lat, max_lon, max_lat


def _with_holes(shape, envelope: _ENVELOPE) -> _ENVELOPE:
    # Holes should lie inside their shape, but nothing enforces it
    if not shape.holes:
        return envelope
    return _union(envelope, *(shape_envelope(hole, pad=False) for hole in shape.holes))


_ENVELOPE_FUNCS: Dict[Type, Callable[..., _ENVELOPE]] = {
    GeoPoint: lambda x: (
        x.coordinate.longitude, x.coordinate.latitude,
        x.coordinate.longitude, x.coordinate.latitude,
    ),
    GeoLineString: lambda x: _vertex_envelope(x.vertices),
    GeoPolygon: lambda x: _with_holes(x, _vertex_envelope(x.outline)),
    GeoBox: lambda x: _with_holes(x, _vertex_envelope(x.bounding_coords())),
    GeoCircle: lambda x: _with_holes(x, _cap_envelope(x.center, x.radius)),
    GeoEllipse: lambda x: _with_holes(
        x, _cap_envelope(x.center, max(x.semi_major, x.semi_minor))
    ),
    GeoRing: lambda x: _with_holes(x, _cap_envelope(x.center, x.outer_radius)),
    MultiGeoPoint: lambda x: _union(*(shape_envelope(s, pad=False) for s in x.geoshapes)),
    MultiGeoLineString: lambda x: _union(*(shape_envelope(s, pad=False) for s in x.geoshapes)),
    MultiGeoPolygon: lambda x: _union(*(shape_envelope(s, pad=False) for s in x.geoshapes)),
}


def shape_envelope(shape: BaseShape, pad: bool = True) -> _ENVELOPE:
    """
    A (min_lon, min_lat, max_lon, max_lat) box guaranteed to contain everything
    the exact predicates consider part of the shape; two shapes whose envelopes
    don't overlap can't intersect. Unlike shape.bounds (which for curved shapes
    is drawn through the cardinal points) this never undercuts the shape, so it
    is safe for discarding candidates. Unrecognized shape types span the globe.
    """
    func = _ENVELOPE_FUNCS.get(type(shape))
    if func is None:
        return -math.inf, -math.inf, math.inf, math.inf

    min_lon, min_lat, max_lon, max_lat = func(shape)
    if not pad:
        return min_lon, min_lat, max_lon, max_lat
    return (
        min_lon - _ENVELOPE_PAD, min_lat - _ENVELOPE_PAD,
        max_lon + _ENVELOPE_PAD, max_lat + _ENVELOPE_PAD
    )


def _points_in_ring(
    ring: Sequence[Coordinate],
    lons: np.ndarray,
    lats: np.ndarray
) -> np.ndarray:
    """
    Vectorized GeoPolygon._point_in_polygon (with include_boundary=False).
    Every expression is evaluated in the same order as the scalar version so
    the float results are identical.
    """
    on_edge = np.zeros(len(lons), dtype=bool)
    crossings = np.zeros(len(lons), dtype=bool)
    for edge in zip(ring, [*ring[1:], ring[0]]):
        start, end = ensure_edge_bounds(edge[0], edge[1])
        lat_a, lat_b = start.latitude, end.latitude
        lon_a, lon_b = start.longitude, end.longitude

        mid_lon = (lon_a + lon_b) / 2
        p_lon = np.where(
            lons - mid_lon > 180,
            lons - 360,
            np.where(mid_lon - lons > 180, lons + 360, lons)
        )

        if lat_a == lat_b:
            on_edge |= (
                (lats == lat_a) & (min(lon_a, lon_b) <= p_lon) & (p_lon <= max(lon_a, lon_b))
            )
            continue

        lon_cross = lon_a + (lats - lat_a) * (lon_b - lon_a) / (lat_b - lat_a)

        diff = np.abs(p_lon - lon_cross)
        is_close = (
            (lon_cross == p_lon)
            | (diff <= np.abs(_ISCLOSE_REL_TOL * p_lon))
            | (diff <= np.abs(_ISCLOSE_REL_TOL * lon_cross))
            | (diff <= COORDINATE_TOLERANCE)
        )
        on_edge |= (min(lat_a, lat_b) <= lats) & (lats <= max(lat_a, lat_b)) & is_close

        crossings ^= ((lat_a > lats) != (lat_b > lats)) & (lon_cross > p_lon)

    return crossings & ~on_edge


def _exclude_holes(shape, result: np.ndarray, lons: np.ndarray, lats: np.ndarray):
    for hole in shape.holes:
        idx = np.flatnonzero(result)
        if not len(idx):
            break

        in_hole = points_in_shape(hole, lons[idx], lats[idx])
        if in_hole is None:
            return None
        result[idx] = ~in_hole

    return result


def _polygon_contains(polygon: GeoPolygon, lons: np.ndarray, lats: np.ndarray):
    """Vectorized GeoPolygon.contains_coordinate"""
    min_lon, min_lat, max_lon, max_lat = polygon.bounds
    result = (min_lon <= lons) & (lons <= max_lon) & (min_lat <= lats) & (lats <= max_lat)
    idx = np.flatnonzero(result)
    if len(idx):
        result[idx] = _points_in_ring(polygon.outline, lons[idx], lats[idx])

    return _exclude_holes(polygon, result, lons, lats)


def _box_contains(box: GeoBox, lons: np.ndarray, lats: np.ndarray):
    """Vectorized GeoBox.contains_coordinate"""
    result = (
        (box.nw_bound.longitude <= lons) & (lons <= box.se_bound.longitude)
        & (box.se_bound.latitude <= lats) & (lats <= box.nw_bound.latitude)
    )
    return _exclude_holes(box, result, lons, lats)


_POINT_KERNELS: Dict[Type, Callable[..., Optional[np.ndarray]]] = {
    GeoPolygon: _polygon_contains,
    GeoBox: _box_contains,
}


def points_in_shape(shape: BaseShape, lons: np.ndarray, lats: np.ndarray) -> Optional[np.ndarray]:
    """
    Vectorized shape.contains_coordinate over many points, or None if the shape
    (or one of its holes) has no vectorized implementation.
    """
    kernel = _POINT_KERNELS.get(type(shape))
    if kernel is None:
        return None
    return kernel(shape, lons, lats)


class ShapeColumns:
    """
    Per-shape arrays derived from a sequence of shapes. Each group of arrays is
    computed the first time it is needed, and subsets created with take() carry
    over whatever has already been computed.
    """

    def __init__(self, shapes: Sequence[BaseShape]):
        self._shapes = shapes

    def __len__(self):
        return len(self._shapes)

    @cached_property
    def envelopes(self) -> np.ndarray:
        """(N, 4) array of shape envelopes; see shape_envelope"""
        # Points (usually the bulk of a large collection) come straight from
        # their coordinates; everything else goes through shape_envelope
        is_point, coords = self._points
        envelopes = np.empty((len(self._shapes), 4))
        envelopes[:, :2] = coords - _ENVELOPE_PAD
        envelopes[:, 2:] = coords + _ENVELOPE_PAD
        for i in np.flatnonzero(~is_point):
            envelopes[i] = shape_envelope(self._shapes[int(i)])
        return envelopes

    @cached_property
    def _times(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Built as lists and converted once; per-element numpy writes are slow
        starts, ends, has_dt = [], [], []
        for shape in self._shapes:
            dt = shape.dt
            if dt is None:
                starts.append(0)
                ends.append(0)
                has_dt.append(False)
            else:
                starts.append(_to_micros(dt.start))
                ends.append(_to_micros(dt.end))
                has_dt.append(True)
        return (
            np.array(starts, dtype=np.int64),
            np.array(ends, dtype=np.int64),
            np.array(has_dt, dtype=bool),
        )

    @property
    def dt_start(self) -> np.ndarray:
        """Shape start times, in microseconds since the epoch (0 where has_dt is False)"""
        return self._times[0]

    @property
    def dt_end(self) -> np.ndarray:
        """Shape end times, in microseconds since the epoch (0 where has_dt is False)"""
        return self._times[1]

    @property
    def has_dt(self) -> np.ndarray:
        """Whether each shape carries time information"""
        return self._times[2]

    @cached_property
    def _points(self) -> Tuple[np.ndarray, np.ndarray]:
        is_point, coords = [], []
        for shape in self._shapes:
            if type(shape) is GeoPoint:  # pylint: disable=unidiomatic-typecheck
                is_point.append(True)
                coords.append((shape.coordinate.longitude, shape.coordinate.latitude))
            else:
                is_point.append(False)
                coords.append((math.nan, math.nan))
        return (
            np.array(is_point, dtype=bool),
            np.array(coords, dtype=np.float64).reshape(len(self._shapes), 2),
        )

    @property
    def is_point(self) -> np.ndarray:
        """Whether each shape is a GeoPoint"""
        return self._points[0]

    @property
    def point_coords(self) -> np.ndarray:
        """(N, 2) array of GeoPoint longitudes/latitudes (NaN for other shapes)"""
        return self._points[1]

    def take(self, idx: np.ndarray, shapes: Sequence[BaseShape]) -> 'ShapeColumns':
        """
        The columns for a subset of rows. shapes must be the corresponding
        subset of this object's shapes, in the same order as idx.
        """
        new = ShapeColumns(shapes)
        if 'envelopes' in self.__dict__:
            new.__dict__['envelopes'] = self.envelopes[idx]
        if '_times' in self.__dict__:
            new.__dict__['_times'] = tuple(arr[idx] for arr in self._times)
        if '_points' in self.__dict__:
            new.__dict__['_points'] = tuple(arr[idx] for arr in self._points)
        return new

    def envelope_overlaps(self, envelope: _ENVELOPE) -> np.ndarray:
        """Rows whose envelopes overlap the given envelope"""
        env = self.envelopes
        return (
            (env[:, 0] <= envelope[2]) & (env[:, 2] >= envelope[0])
            & (env[:, 1] <= envelope[3]) & (env[:, 3] >= envelope[1])
        )

    def time_intersects(self, dt: TimeInterval) -> np.ndarray:
        """
        Rows whose time intersects dt, mirroring TimeInterval.intersects
        (right-open intervals, with instants special-cased). False for rows
        without time information.
        """
        q_start, q_end = _to_micros(dt.start), _to_micros(dt.end)
        starts, ends, has_dt = self._times
        row_instant = starts == ends
        if q_start == q_end:
            hits = np.where(row_instant, starts == q_start, (starts <= q_start) & (q_start < ends))
        else:
            hits = np.where(
                row_instant,
                (q_start <= starts) & (starts < q_end),
                (starts < q_end) & (q_start < ends)
            )
        return hits & has_dt

    def time_within(self, dt: TimeInterval) -> np.ndarray:
        """Rows whose time lies within dt (TimeInterval.issubset). False for rows without time."""
        q_start, q_end = _to_micros(dt.start), _to_micros(dt.end)
        starts, ends, has_dt = self._times
        return (q_start <= starts) & (ends <= q_end) & has_dt

    def time_contains(self, dt: TimeInterval) -> np.ndarray:
        """Rows whose time contains dt (TimeInterval.issuperset). False for rows without time."""
        q_start, q_end = _to_micros(dt.start), _to_micros(dt.end)
        starts, ends, has_dt = self._times
        return (starts <= q_start) & (q_end <= ends) & has_dt

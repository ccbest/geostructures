"""
Columnar (array) views of shape collections.

Collections derive these arrays from their shapes and use them to answer
whole-collection questions - time and spatial filtering - with numpy rather
than one shape at a time. They are purely an acceleration structure: every
result computed here must match what the per-shape methods would return, and
the tests compare the two directly.
"""

__all__ = ['ShapeColumns', 'shape_envelope', 'points_in_shape']

from datetime import datetime
from functools import cached_property
import math
from typing import Callable, Dict, Optional, Sequence, Tuple, Type

import numpy as np

from geostructures import geodesic
from geostructures._base import BaseShape
from geostructures._const import COORDINATE_TOLERANCE, EARTH_RADIUS_METERS
from geostructures._geometry import ensure_edge_bounds
from geostructures.coordinates import Coordinate
from geostructures.multistructures import MultiGeoLineString, MultiGeoPoint, MultiGeoPolygon
from geostructures.structures import (
    GeoBox, GeoCircle, GeoEllipse, GeoLineString, GeoPoint, GeoPolygon, GeoRing
)
from geostructures.time import TimeInterval, _endpoint, _TimeColumns


_ENVELOPE = Tuple[float, float, float, float]

# Envelopes are padded well beyond the tolerance the exact predicates allow
# when deciding whether edges touch
_ENVELOPE_PAD = 10 * COORDINATE_TOLERANCE

# Curved shapes are enveloped by a spherical cap. Ellipsoidal geodesic
# algorithms (vincenty, karney) can reach ~0.6% further in latitude than the
# spherical earth, so the cap radius is inflated to cover them
_CAP_RADIUS_FACTOR = 1.01

# Mirrors math.isclose's default, used by GeoPolygon._point_in_polygon
_ISCLOSE_REL_TOL = 1e-9

# The curved-shape kernels use numpy trigonometry, which can differ from
# math's in the last bit. Points this close to a threshold are re-tested with
# the shape's own contains_coordinate, so the results match it exactly
_DISTANCE_MARGIN = 1e-3  # meters
_RELATIVE_MARGIN = 1e-9
_BEARING_MARGIN = 1e-6  # degrees

# Bearings are ill-conditioned close to the center, and distances and bearings
# both close to the antipode; points there are always re-tested
_BEARING_MIN_DISTANCE = 10.  # meters
_ANTIPODE_DISTANCE = math.pi * EARTH_RADIUS_METERS - 1_000.  # meters


def _micros(dt: datetime) -> int:
    """Microseconds since the epoch, as TimeInterval stores them (naive datetimes are UTC)"""
    return _endpoint(dt)[0]


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
    don't overlap can't intersect. It is kept apart from shape.bounds because it
    is padded beyond the predicates' tolerance, and because a shape covering a
    pole or crossing the antimeridian spans every longitude here, so it is safe
    for discarding candidates. Unrecognized shape types span the globe.

    The padded envelope is cached on the shape (and discarded with its other
    cached values), so collections can gather it rather than recompute it.
    """
    if pad:
        cached = shape.__dict__.get('_envelope')
        if cached is not None:
            return cached

    func = _ENVELOPE_FUNCS.get(type(shape))
    if func is None:
        return -math.inf, -math.inf, math.inf, math.inf

    min_lon, min_lat, max_lon, max_lat = func(shape)
    if not pad:
        return min_lon, min_lat, max_lon, max_lat
    envelope = (
        min_lon - _ENVELOPE_PAD, min_lat - _ENVELOPE_PAD,
        max_lon + _ENVELOPE_PAD, max_lat + _ENVELOPE_PAD
    )
    shape.__dict__['_envelope'] = envelope
    return envelope


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
        if idx.size == 0:
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


def _haversine_is_active() -> bool:
    """The curved kernels mirror the Haversine functions, so apply only while they are in use"""
    return (
        geodesic._active_distance is geodesic.haversine_distance  # pylint: disable=protected-access
        and geodesic._active_bearing is geodesic.haversine_bearing  # pylint: disable=protected-access
    )


def _haversine_distances(start_lon, start_lat, end_lon, end_lat) -> np.ndarray:
    """Vectorized geodesic.haversine_distance, operation for operation"""
    lon1, lat1 = np.radians(start_lon), np.radians(start_lat)
    lon2, lat2 = np.radians(end_lon), np.radians(end_lat)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return EARTH_RADIUS_METERS * c


def _haversine_bearings(start_lon, start_lat, end_lon, end_lat) -> np.ndarray:
    """Vectorized geodesic.haversine_bearing, operation for operation"""
    lon1, lat1 = np.radians(start_lon), np.radians(start_lat)
    lon2, lat2 = np.radians(end_lon), np.radians(end_lat)

    dlon = lon2 - lon1

    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

    return (np.degrees(np.arctan2(y, x)) + 360) % 360


def _distance_margin(radius: float) -> float:
    return _DISTANCE_MARGIN + _RELATIVE_MARGIN * radius


def _near_bearing(bearings: np.ndarray, angle: float) -> np.ndarray:
    """Bearings within _BEARING_MARGIN of angle, either way around the circle"""
    diff = (bearings - angle) % 360
    return (diff <= _BEARING_MARGIN) | (diff >= 360 - _BEARING_MARGIN)


def _unreliable(distances: np.ndarray) -> np.ndarray:
    """Points whose distance or bearing from the center is ill-conditioned"""
    return (distances < _BEARING_MIN_DISTANCE) | (distances > _ANTIPODE_DISTANCE)


def _resolve_curved(shape, inside: np.ndarray, unsure: np.ndarray, lons: np.ndarray, lats: np.ndarray):
    """
    Excludes holes from the points decided on arrays, then re-tests the points
    too close to call with the shape's own contains_coordinate.
    """
    result = _exclude_holes(shape, inside & ~unsure, lons, lats)
    if result is None:
        return None

    for i in np.flatnonzero(unsure).tolist():
        result[i] = shape.contains_coordinate(Coordinate(float(lons[i]), float(lats[i])))
    return result


def _circle_contains(circle: GeoCircle, lons: np.ndarray, lats: np.ndarray):
    """Vectorized GeoCircle.contains_coordinate (Haversine only)"""
    if not _haversine_is_active():
        return None

    center = circle.center
    distances = _haversine_distances(lons, lats, center.longitude, center.latitude)
    unsure = (
        (np.abs(distances - circle.radius) <= _distance_margin(circle.radius))
        | (distances > _ANTIPODE_DISTANCE)
    )
    return _resolve_curved(circle, distances <= circle.radius, unsure, lons, lats)


def _ellipse_contains(ellipse: GeoEllipse, lons: np.ndarray, lats: np.ndarray):
    """Vectorized GeoEllipse.contains_coordinate (Haversine only)"""
    if not _haversine_is_active():
        return None

    semi_major, semi_minor = ellipse.semi_major, ellipse.semi_minor
    longest, shortest = max(semi_major, semi_minor), min(semi_major, semi_minor)
    if shortest <= 0:
        return None

    center = ellipse.center
    bearings = _haversine_bearings(center.longitude, center.latitude, lons, lats)
    angles = np.radians(bearings - ellipse.rotation)
    radii = (
        semi_major
        * semi_minor
        / np.sqrt(
            (semi_major**2) * (np.sin(angles) ** 2)
            + (semi_minor**2) * (np.cos(angles) ** 2)
        )
    )
    distances = _haversine_distances(center.longitude, center.latitude, lons, lats)

    # The radius varies with bearing, so a bearing's rounding error moves it
    # by up to roughly longest * (longest / shortest) times that error
    margin = _distance_margin(longest * (longest / shortest) ** 2)
    unsure = (np.abs(distances - radii) <= margin) | _unreliable(distances)
    return _resolve_curved(ellipse, distances <= radii, unsure, lons, lats)


def _ring_contains(ring: GeoRing, lons: np.ndarray, lats: np.ndarray):
    """Vectorized GeoRing.contains_coordinate (Haversine only)"""
    if not _haversine_is_active():
        return None

    center = ring.center
    distances = _haversine_distances(center.longitude, center.latitude, lons, lats)
    inside = (ring.inner_radius <= distances) & (distances <= ring.outer_radius)
    unsure = (
        (np.abs(distances - ring.inner_radius) <= _distance_margin(ring.inner_radius))
        | (np.abs(distances - ring.outer_radius) <= _distance_margin(ring.outer_radius))
        | (distances > _ANTIPODE_DISTANCE)
    )

    if ring.angle_max - ring.angle_min < 360:
        bearings = _haversine_bearings(center.longitude, center.latitude, lons, lats)
        inside &= (bearings - ring.angle_min) % 360 <= ring.angle_max - ring.angle_min
        unsure |= (
            _near_bearing(bearings, ring.angle_min)
            | _near_bearing(bearings, ring.angle_max)
            | _unreliable(distances)
        )

    return _resolve_curved(ring, inside, unsure, lons, lats)


_POINT_KERNELS: Dict[Type, Callable[..., Optional[np.ndarray]]] = {
    GeoPolygon: _polygon_contains,
    GeoBox: _box_contains,
    GeoCircle: _circle_contains,
    GeoEllipse: _ellipse_contains,
    GeoRing: _ring_contains,
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

    def bounds(self) -> _ENVELOPE:
        """
        The (min_lon, min_lat, max_lon, max_lat) covering every shape's bounds,
        as combining each shape.bounds would give. Points come from their
        coordinate arrays; only other shapes are visited.
        """
        is_point, coords = self._points
        all_bounds = [self._shapes[i].bounds for i in np.flatnonzero(~is_point).tolist()]
        if is_point.any():
            # Rows for other shapes hold NaN, which fmin and fmax ignore
            lons, lats = coords[:, 0], coords[:, 1]
            all_bounds.append((
                float(np.fmin.reduce(lons)), float(np.fmin.reduce(lats)),
                float(np.fmax.reduce(lons)), float(np.fmax.reduce(lats)),
            ))

        return (
            min(x[0] for x in all_bounds),
            min(x[1] for x in all_bounds),
            max(x[2] for x in all_bounds),
            max(x[3] for x in all_bounds)
        )

    @cached_property
    def _times(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Start and end microseconds (zero for rows without time), and whether each row has time"""
        times = _TimeColumns.from_intervals([x.dt for x in self._shapes])
        return times.starts, times.ends, times.has_dt

    @property
    def has_dt(self) -> np.ndarray:
        """Whether each shape carries time information"""
        return self._times[2]

    def starts_at_or_after(self, dt: datetime) -> np.ndarray:
        """Rows whose start time is at or after dt. False for rows without time."""
        return (self._times[0] >= _micros(dt)) & self.has_dt

    def ends_before(self, dt: datetime) -> np.ndarray:
        """Rows whose end time is before dt. False for rows without time."""
        return (self._times[1] < _micros(dt)) & self.has_dt

    def start_order(self) -> np.ndarray:
        """Row indices ordered by start time; stable, like sorted()"""
        return np.argsort(self._times[0], kind='stable')

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
            starts, ends, has_dt = self._times
            new.__dict__['_times'] = starts[idx], ends[idx], has_dt[idx]
        if '_points' in self.__dict__:
            new.__dict__['_points'] = tuple(arr[idx] for arr in self._points)
        return new

    def concat(self, other: 'ShapeColumns', shapes: Sequence[BaseShape]) -> 'ShapeColumns':
        """
        The columns for self's rows followed by other's. shapes must be the
        corresponding concatenation of the two objects' shapes. Groups already
        computed for self are carried over (computing them for other).
        """
        new = ShapeColumns(shapes)
        for name in ('envelopes', '_times', '_points'):
            if name not in self.__dict__:
                continue
            mine, theirs = getattr(self, name), getattr(other, name)
            if isinstance(mine, np.ndarray):
                new.__dict__[name] = np.concatenate([mine, theirs])
            else:
                new.__dict__[name] = tuple(np.concatenate([a, b]) for a, b in zip(mine, theirs))
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
        q_start, q_end = dt._start, dt._end  # pylint: disable=protected-access
        starts, ends, has_dt = self._times
        row_instant = starts == ends
        if q_start == q_end:
            hits = np.where(row_instant, starts == q_start, (starts <= q_start) & (ends > q_start))
        else:
            hits = np.where(
                row_instant,
                (starts >= q_start) & (starts < q_end),
                (starts < q_end) & (ends > q_start)
            )
        return hits & has_dt

    def time_within(self, dt: TimeInterval) -> np.ndarray:
        """Rows whose time lies within dt (TimeInterval.issubset). False for rows without time."""
        q_start, q_end = dt._start, dt._end  # pylint: disable=protected-access
        starts, ends, has_dt = self._times
        return (starts >= q_start) & (ends <= q_end) & has_dt

    def time_contains(self, dt: TimeInterval) -> np.ndarray:
        """Rows whose time contains dt (TimeInterval.issuperset). False for rows without time."""
        q_start, q_end = dt._start, dt._end  # pylint: disable=protected-access
        starts, ends, has_dt = self._times
        return (starts <= q_start) & (ends >= q_end) & has_dt

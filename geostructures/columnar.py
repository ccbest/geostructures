"""
Array-backed (columnar) collections of geostructures.

A FeatureCollection holds one Python object per shape, which is convenient but
costs a few microseconds and a few hundred bytes for every point. A
PointCollection instead stores its points as parallel numpy arrays (longitude,
latitude, optional Z/M, time bounds and one array per property), so building,
filtering and bounding millions of points runs at numpy speed.

GeoPoint objects are only created when asked for: by integer indexing,
iteration, or the to_points()/to_feature_collection()/to_track() conversions.
Filtering returns a new PointCollection.

Results match the equivalent FeatureCollection methods. Where a shape has no
vectorized containment test (e.g. GeoEllipse, GeoRing, GeoLineString), the
shape is asked about each point's Coordinate in turn, which runs at about the
speed of the same filter on a FeatureCollection.
"""

from __future__ import annotations

__all__ = ['PointCollection']

from datetime import datetime, timezone
import math
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union, cast

import numpy as np
import numpy.typing as npt

from geostructures._base import BaseShape, MultiShapeBase
from geostructures._vectorized import bound_coordinates, coordinate_equals_mask, shape_contains_mask
from geostructures.collections import FeatureCollection, Track
from geostructures.coordinates import Coordinate
from geostructures.structures import GeoLineString, GeoPoint, PolygonBase
from geostructures.time import TimeInterval
from geostructures.utils.conditional_imports import import_optional


# Time bounds are stored as UTC microseconds, matching datetime's resolution
_TIME_DTYPE = 'datetime64[us]'

# The property names GeoPoint.properties reports a point's time bounds under
_DT_START_KEY = 'datetime_start'
_DT_END_KEY = 'datetime_end'

# The library's contains_shape() implementations that, given a point, test
# only contains_coordinate(point.centroid). Shapes using one of these can be
# asked about bare Coordinates rather than full GeoPoints.
_COORDINATE_CONTAINS_SHAPE = (
    PolygonBase.contains_shape,
    GeoLineString.contains_shape,
    GeoPoint.contains_shape,
    MultiShapeBase.contains_shape,
)


def _arrays_equal(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> bool:
    """Element-wise equality of two optional arrays, treating NaN/NaT as equal to themselves"""
    if a is None or b is None:
        return a is b

    if a.shape != b.shape:
        return False

    if a.dtype.kind == 'M' and b.dtype.kind == 'M':
        return bool(np.all((a == b) | (np.isnat(a) & np.isnat(b))))

    if a.dtype.kind == 'f' and b.dtype.kind == 'f':
        return bool(np.array_equal(a, b, equal_nan=True))

    return a.tolist() == b.tolist()


def _as_column(values: Any, name: str, size: int) -> np.ndarray:
    """
    Copies a property's values into a one-dimensional array of the given
    length. Arrays keep their dtype. Other sequences become object arrays,
    narrowed to bool/int64/float64 only when every value has that exact type,
    so values always come back out with the type they went in with.
    """
    if isinstance(values, np.ndarray):
        if values.ndim != 1:
            raise ValueError(f"Property '{name}' must be one-dimensional")
        arr = values.copy()
    else:
        # fromiter keeps sequence-valued properties (lists, tuples) as single
        # values, where np.array would turn them into extra dimensions
        items = list(values)
        arr = np.fromiter(items, dtype=object, count=len(items))
        types = set(map(type, items))
        if len(types) == 1 and next(iter(types)) in (bool, int, float):
            try:
                arr = np.array(items, dtype=next(iter(types)))
            except OverflowError:
                pass  # ints too large for int64 stay as Python ints

    if arr.shape[0] != size:
        raise ValueError(f"Property '{name}' has {arr.shape[0]} values; expected {size}")

    return arr


def _as_float_array(values: npt.ArrayLike, name: str, size: Optional[int] = None) -> np.ndarray:
    """Copies values into a one-dimensional float64 array, optionally checking its length"""
    arr = np.array(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f'{name} must be one-dimensional')

    if size is not None and arr.shape[0] != size:
        raise ValueError(f'{name} has {arr.shape[0]} values; expected {size}')

    return arr


def _as_time_array(values: Any, name: str, size: int) -> np.ndarray:
    """
    Copies datetimes (or None) into a UTC datetime64 array. Naive datetimes are
    taken to already be UTC, matching TimeInterval.
    """
    if isinstance(values, np.ndarray) and values.dtype.kind == 'M':
        arr = values.astype(_TIME_DTYPE)
    else:
        arr = np.array(
            [_datetime_to_utc_naive(x) if isinstance(x, datetime) else x for x in values],
            dtype=_TIME_DTYPE
        )

    if arr.ndim != 1 or arr.shape[0] != size:
        raise ValueError(f'{name} must be one-dimensional with {size} values')

    return arr


def _datetime_to_datetime64(dt: datetime) -> np.datetime64:
    return np.datetime64(_datetime_to_utc_naive(dt), 'us')


def _datetime_to_utc_naive(dt: datetime) -> datetime:
    """Converts to naive UTC; naive datetimes are taken to already be UTC, matching TimeInterval"""
    if dt.tzinfo is None:
        return dt

    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _to_python(value: Any) -> Any:
    """Converts numpy scalars to their Python equivalents"""
    return value.item() if isinstance(value, np.generic) else value


class PointCollection:
    """
    A collection of points stored as parallel arrays rather than as one
    GeoPoint per point. See the module docstring for the trade-offs.

    Coordinates are wrapped into range exactly as Coordinate does it. Time
    bounds are stored in UTC; naive datetimes are taken to already be UTC. All
    arrays are copied on construction and exposed read-only, so a collection
    never changes once built.

    Args:
        longitude:
            Longitudes, in degrees

        latitude:
            Latitudes, in degrees

        z: (Optional)
            Z values. NaN marks a point without one.

        m: (Optional)
            M values. NaN marks a point without one.

        dt_start: (Optional)
            Start times, as datetimes (None for a point without a time) or
            a datetime64 array (NaT for a point without a time)

        dt_end: (Optional)
            End times, in the same form as dt_start. If only one of dt_start
            and dt_end is given, each point's time is an instant.

        properties: (Optional)
            A mapping of property name to that property's values, one per point

    Examples:
        ```python
        points = PointCollection(lons, lats, dt_start=times)
        in_area = points.filter_contained_by(polygon)
        red = points[points.properties['color'] == 'red']
        ```
    """

    # Eight parallel arrays/mappings are the point of this class
    # pylint: disable=too-many-instance-attributes

    __slots__ = ('_lon', '_lat', '_z', '_m', '_start', '_end', '_props', '_present')

    # Collections compare by value but aren't hashable, like lists
    __hash__ = None  # type: ignore[assignment]

    def __init__(
        self,
        longitude: npt.ArrayLike,
        latitude: npt.ArrayLike,
        *,
        z: Optional[npt.ArrayLike] = None,
        m: Optional[npt.ArrayLike] = None,
        dt_start: Optional[Any] = None,
        dt_end: Optional[Any] = None,
        properties: Optional[Mapping[str, Any]] = None,
    ):
        lon = _as_float_array(longitude, 'longitude')
        size = lon.shape[0]
        lat = _as_float_array(latitude, 'latitude', size)
        if not (np.isfinite(lon).all() and np.isfinite(lat).all()):
            raise ValueError('Longitudes and latitudes must be finite')
        bound_coordinates(lon, lat)

        _z = _as_float_array(z, 'z', size) if z is not None else None
        _m = _as_float_array(m, 'm', size) if m is not None else None

        if dt_start is None and dt_end is None:
            start = np.full(size, np.datetime64('NaT', 'us'))
            end = start
        else:
            start = _as_time_array(dt_start if dt_start is not None else dt_end, 'dt_start', size)
            end = _as_time_array(dt_end, 'dt_end', size) if dt_end is not None else start

        if not np.array_equal(np.isnat(start), np.isnat(end)):
            raise ValueError('Every point must have both a start and an end time, or neither')

        if np.any(end < start):
            raise ValueError('End times must not be earlier than start times')

        props = {
            name: _as_column(values, name, size)
            for name, values in (properties or {}).items()
        }

        self._set_arrays(lon, lat, _z, _m, start, end, props, {})

    def __bool__(self) -> bool:
        return bool(len(self))

    def __contains__(self, item: Any) -> bool:
        if not isinstance(item, GeoPoint):
            return False

        return bool(self._equals_mask(item).any())

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, PointCollection):
            return False

        return (
            _arrays_equal(self._lon, other._lon) and
            _arrays_equal(self._lat, other._lat) and
            _arrays_equal(self._z, other._z) and
            _arrays_equal(self._m, other._m) and
            _arrays_equal(self._start, other._start) and
            _arrays_equal(self._end, other._end) and
            self._props.keys() == other._props.keys() and
            all(_arrays_equal(self._props[k], other._props[k]) for k in self._props) and
            self._present.keys() == other._present.keys() and
            all(_arrays_equal(self._present[k], other._present[k]) for k in self._present)
        )

    def __getitem__(self, item: Union[int, slice, Sequence[int], np.ndarray]) -> Any:
        """
        Integer indexing returns a GeoPoint. Slices, integer arrays and boolean
        masks return a new PointCollection.
        """
        if isinstance(item, (int, np.integer)):
            index = int(item)
            if not -len(self) <= index < len(self):
                raise IndexError('PointCollection index out of range')
            return self._point_at(index % len(self))

        if isinstance(item, slice):
            return self._subset(item)

        index_arr = np.asarray(item)
        if index_arr.dtype.kind not in 'biu':
            raise TypeError(f'PointCollections cannot be indexed by {type(item)}')

        return self._subset(index_arr)

    def __iter__(self) -> Iterator[GeoPoint]:
        """Iterates through the points, creating a GeoPoint for each"""
        lons, lats = self._lon.tolist(), self._lat.tolist()
        zs = self._z.tolist() if self._z is not None else [math.nan] * len(self)
        ms = self._m.tolist() if self._m is not None else [math.nan] * len(self)
        starts, ends = self._start.astype(object).tolist(), self._end.astype(object).tolist()
        columns = {name: values.tolist() for name, values in self._props.items()}
        present = {name: mask.tolist() for name, mask in self._present.items()}

        for i, (lon, lat, z, m, start, end) in enumerate(zip(lons, lats, zs, ms, starts, ends)):
            yield self._make_point(
                lon, lat, z, m, start, end,
                {
                    name: values[i] for name, values in columns.items()
                    if name not in present or present[name][i]
                }
            )

    def __len__(self) -> int:
        return int(self._lon.shape[0])

    def __repr__(self) -> str:
        if not self:
            return '<Empty PointCollection>'

        return f'<PointCollection with {len(self)} points>'

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """The (min longitude, min latitude, max longitude, max latitude) of the points"""
        if not self:
            raise ValueError('Cannot compute the bounds of an empty collection.')

        return (
            float(self._lon.min()), float(self._lat.min()),
            float(self._lon.max()), float(self._lat.max())
        )

    @property
    def centroid(self) -> Coordinate:
        """The average longitude and latitude of the points"""
        if not self:
            raise ValueError('Cannot compute the centroid of an empty collection.')

        return Coordinate(float(np.average(self._lon)), float(np.average(self._lat)))

    @property
    def dt_end(self) -> np.ndarray:
        """End times, as UTC datetime64 (NaT for points without a time). Read-only."""
        return self._end

    @property
    def dt_start(self) -> np.ndarray:
        """Start times, as UTC datetime64 (NaT for points without a time). Read-only."""
        return self._start

    @property
    def geospan(self) -> float:
        """The width plus the height of self.bounds, in degrees"""
        bounds = self.bounds
        return bounds[2] - bounds[0] + bounds[3] - bounds[1]

    @property
    def latitude(self) -> np.ndarray:
        """Latitudes, in degrees. Read-only."""
        return self._lat

    @property
    def longitude(self) -> np.ndarray:
        """Longitudes, in degrees. Read-only."""
        return self._lon

    @property
    def m(self) -> Optional[np.ndarray]:
        """M values (NaN for points without one), or None if no point has one. Read-only."""
        return self._m

    @property
    def properties(self) -> Mapping[str, np.ndarray]:
        """
        Property arrays, by name. A point that doesn't carry a property holds
        None in that property's array. Read-only.
        """
        return MappingProxyType(self._props)

    @property
    def z(self) -> Optional[np.ndarray]:
        """Z values (NaN for points without one), or None if no point has one. Read-only."""
        return self._z

    def _equals_mask(self, point: GeoPoint) -> np.ndarray:
        """Vectorized GeoPoint.__eq__ against a single point"""
        mask = coordinate_equals_mask(self._lon, self._lat, self._z, self._m, point.coordinate)
        if point.dt is None:
            return mask & np.isnat(self._start)

        return (
            mask &
            (self._start == _datetime_to_datetime64(point.dt.start)) &
            (self._end == _datetime_to_datetime64(point.dt.end))
        )

    @classmethod
    def _from_arrays(
        cls,
        lon: np.ndarray,
        lat: np.ndarray,
        z: Optional[np.ndarray],
        m: Optional[np.ndarray],
        start: np.ndarray,
        end: np.ndarray,
        props: Dict[str, np.ndarray],
        present: Dict[str, np.ndarray],
    ) -> 'PointCollection':
        """Builds a collection from arrays that are already validated, skipping __init__"""
        collection = cls.__new__(cls)
        collection._set_arrays(lon, lat, z, m, start, end, props, present)
        return collection

    @staticmethod
    def _make_point(
        lon: float,
        lat: float,
        z: Optional[float],
        m: Optional[float],
        start: Optional[datetime],
        end: Optional[datetime],
        properties: Dict[str, Any],
    ) -> GeoPoint:
        """Creates a GeoPoint from one point's (Python-typed) values"""
        coord = Coordinate(
            lon, lat,
            z=None if z is None or math.isnan(z) else z,
            m=None if m is None or math.isnan(m) else m,
            _bounded=False,  # Already wrapped into range on construction
        )
        dt: Optional[Union[datetime, TimeInterval]] = None
        if start is not None and end is not None:
            start, end = start.replace(tzinfo=timezone.utc), end.replace(tzinfo=timezone.utc)
            dt = start if start == end else TimeInterval(start, end)

        return GeoPoint(coord, dt=dt, properties=properties)

    def _point_at(self, index: int) -> GeoPoint:
        """Creates the GeoPoint at the given (non-negative) position"""
        return self._make_point(
            float(self._lon[index]),
            float(self._lat[index]),
            float(self._z[index]) if self._z is not None else None,
            float(self._m[index]) if self._m is not None else None,
            self._start[index].astype(object),
            self._end[index].astype(object),
            {
                name: _to_python(values[index]) for name, values in self._props.items()
                if name not in self._present or self._present[name][index]
            }
        )

    def _property_values(self, name: str) -> np.ndarray:
        """
        A property's values, including the datetime_start/datetime_end
        properties that GeoPoint.properties reports for points with a time.
        Raises KeyError if any point doesn't carry the property.
        """
        if name in self._props:
            if name in self._present and not self._present[name].all():
                raise KeyError(f"Property '{name}' not found in shape properties")
            return self._props[name]

        if name in (_DT_START_KEY, _DT_END_KEY):
            times = self._start if name == _DT_START_KEY else self._end
            if np.isnat(times).any():
                raise KeyError(f"Property '{name}' not found in shape properties")
            values = np.empty(len(self), dtype=object)
            values[:] = [x.replace(tzinfo=timezone.utc) for x in times.astype(object).tolist()]
            return values

        raise KeyError(f"Property '{name}' not found in shape properties")

    def _set_arrays(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        z: Optional[np.ndarray],
        m: Optional[np.ndarray],
        start: np.ndarray,
        end: np.ndarray,
        props: Dict[str, np.ndarray],
        present: Dict[str, np.ndarray],
    ) -> None:
        """Stores the arrays read-only. Z/M arrays holding only NaN are dropped."""
        if z is not None and np.isnan(z).all():
            z = None
        if m is not None and np.isnan(m).all():
            m = None

        arrays = (lon, lat, z, m, start, end, *props.values(), *present.values())
        for arr in arrays:
            if arr is not None:
                arr.setflags(write=False)

        self._lon, self._lat, self._z, self._m = lon, lat, z, m
        self._start, self._end = start, end
        self._props = props
        # Presence masks, only for properties that some points don't carry
        self._present = present

    def _spatial_mask(self, shape: BaseShape) -> Optional[np.ndarray]:
        """
        Whether shape.contains_shape() holds for each point, ignoring time.
        Vectorized where _vectorized has a kernel for the shape; otherwise the
        shape is asked about bare Coordinates, which costs far less than
        building GeoPoints. Returns None if the shape's contains_shape() may do
        more with a point than test its coordinate (e.g. a user subclass).
        """
        if isinstance(shape, GeoPoint):
            return coordinate_equals_mask(self._lon, self._lat, self._z, self._m, shape.coordinate)

        mask = shape_contains_mask(shape, self._lon, self._lat)
        if mask is not None:
            return mask

        if type(shape).contains_shape not in _COORDINATE_CONTAINS_SHAPE:
            return None

        lons, lats = self._lon.tolist(), self._lat.tolist()
        zs = self._z.tolist() if self._z is not None else [math.nan] * len(self)
        ms = self._m.tolist() if self._m is not None else [math.nan] * len(self)
        coords = (
            Coordinate(
                lon, lat,
                z=None if math.isnan(z) else z,
                m=None if math.isnan(m) else m,
                _bounded=False,  # Already wrapped into range on construction
            )
            for lon, lat, z, m in zip(lons, lats, zs, ms)
        )
        return np.fromiter((shape.contains_coordinate(c) for c in coords), dtype=np.bool_, count=len(self))

    def _subset(self, index: Union[slice, np.ndarray]) -> 'PointCollection':
        """Creates a new collection from the points selected by a slice, integer array or mask"""
        return self._from_arrays(
            self._lon[index],
            self._lat[index],
            self._z[index] if self._z is not None else None,
            self._m[index] if self._m is not None else None,
            self._start[index],
            self._end[index],
            {name: values[index] for name, values in self._props.items()},
            {name: mask[index] for name, mask in self._present.items()},
        )

    def _time_contained_mask(self, dt: TimeInterval) -> np.ndarray:
        """Vectorized `point.dt in dt` (closed at both ends, per TimeInterval.issuperset)"""
        q_start, q_end = _datetime_to_datetime64(dt.start), _datetime_to_datetime64(dt.end)
        return (q_start <= self._start) & (self._end <= q_end)

    def filter_by_dt(self, dt: Union[datetime, TimeInterval]) -> 'PointCollection':
        """
        Subsets to the points whose times intersect the given datetime or time
        interval. Points without a time are excluded.

        Args:
            dt:
                A datetime or TimeInterval

        Returns:
            PointCollection
        """
        return self._subset(self.mask_by_dt(dt))

    def filter_by_intersection(self, shape: BaseShape) -> 'PointCollection':
        """
        Subsets to the points that intersect a geoshape, which is optionally
        time-bounded.

        Args:
            shape:
                A geoshape

        Returns:
            PointCollection
        """
        return self._subset(self.mask_by_intersection(shape))

    def filter_by_property(  # pylint: disable=redefined-builtin
        self, property: str, func: Callable[[Any], bool]
    ) -> 'PointCollection':
        """
        Subsets to the points for which func returns True when given the
        point's value of a property. func is called once per point; for large
        collections, a vectorized comparison is much faster:
        `points[points.properties['color'] == 'red']`.

        Args:
            property:
                The property to filter by

            func:
                The function to apply to each point's property value, e.g.
                lambda x: x == 'red'

        Returns:
            PointCollection
        """
        values = self._property_values(property).tolist()
        return self._subset(np.fromiter((bool(func(x)) for x in values), dtype=np.bool_, count=len(values)))

    def filter_contained_by(self, shape: BaseShape) -> 'PointCollection':
        """
        Subsets to the points contained by a geoshape, which is optionally
        time-bounded.

        Args:
            shape:
                A geoshape

        Returns:
            PointCollection
        """
        return self._subset(self.mask_contained_by(shape))

    @classmethod
    def from_geopandas(
        cls,
        df,
        time_start_field: str = _DT_START_KEY,
        time_end_field: str = _DT_END_KEY,
    ) -> 'PointCollection':
        """
        Creates a PointCollection from a geopandas dataframe of points, without
        creating a Python object per point. Associates start and end times to
        the points, if present, and stores the remaining columns as properties.

        Args:
            df:
                A GeoPandas dataframe whose geometries are all Points

            time_start_field:
                The field name for the start time

            time_end_field:
                The field name for the end time. Where a point has only one of
                the two times, that time is used for both.

        Returns:
            PointCollection
        """
        import_optional('pandas')
        import pandas as pd

        geometry = df.geometry
        if not (geometry.geom_type == 'Point').all():
            raise ValueError('PointCollection.from_geopandas requires every geometry to be a Point')

        def _utc_times(field: str) -> Optional[np.ndarray]:
            if field not in df.columns:
                return None
            times = pd.to_datetime(df[field], utc=True).dt.tz_localize(None)
            return times.to_numpy().astype(_TIME_DTYPE)

        start, end = _utc_times(time_start_field), _utc_times(time_end_field)
        if start is not None and end is not None:
            # A point with only one of the two times is an instant at that time
            start, end = np.where(np.isnat(start), end, start), np.where(np.isnat(end), start, end)

        prop_fields = [
            x for x in df.columns
            if x not in (time_start_field, time_end_field, geometry.name)
        ]
        return cls(
            geometry.x.to_numpy(),
            geometry.y.to_numpy(),
            z=geometry.z.to_numpy() if geometry.has_z.any() else None,
            dt_start=start,
            dt_end=end,
            properties={name: df[name].to_numpy() for name in prop_fields},
        )

    @classmethod
    def from_points(cls, points: Iterable[Union[GeoPoint, Coordinate]]) -> 'PointCollection':
        """
        Creates a PointCollection from GeoPoints (or bare Coordinates, which
        have no time or properties), e.g. the points of a FeatureCollection.

        Args:
            points:
                An iterable of GeoPoints and/or Coordinates

        Returns:
            PointCollection
        """
        lons: List[float] = []
        lats: List[float] = []
        zs: List[float] = []
        ms: List[float] = []
        starts: List[Optional[datetime]] = []
        ends: List[Optional[datetime]] = []
        rows: List[Dict[str, Any]] = []

        for point in points:
            if isinstance(point, GeoPoint):
                coord, dt, point_props = point.coordinate, point.dt, point._properties
            elif isinstance(point, Coordinate):
                coord, dt, point_props = point, None, {}
            else:
                raise TypeError(f'PointCollections can only hold points; got {type(point).__name__}')

            lons.append(coord.longitude)
            lats.append(coord.latitude)
            zs.append(math.nan if coord.z is None else coord.z)
            ms.append(math.nan if coord.m is None else coord.m)
            starts.append(None if dt is None else _datetime_to_utc_naive(dt.start))
            ends.append(None if dt is None else _datetime_to_utc_naive(dt.end))
            rows.append(point_props)

        size = len(lons)
        names = list(dict.fromkeys(name for row in rows for name in row))
        props: Dict[str, np.ndarray] = {}
        present: Dict[str, np.ndarray] = {}
        for name in names:
            mask = np.fromiter((name in row for row in rows), dtype=np.bool_, count=size)
            values = [row.get(name) for row in rows]
            props[name] = _as_column(values, name, size)
            if not mask.all():
                present[name] = mask

        return cls._from_arrays(
            np.array(lons, dtype=np.float64),
            np.array(lats, dtype=np.float64),
            np.array(zs, dtype=np.float64),
            np.array(ms, dtype=np.float64),
            np.array(starts, dtype=_TIME_DTYPE),
            np.array(ends, dtype=_TIME_DTYPE),
            props,
            present,
        )

    def intersects(self, shape: BaseShape) -> bool:
        """
        Whether any point intersects a geoshape, which is optionally time-bounded.

        Args:
            shape:
                A geoshape

        Returns:
            bool
        """
        return bool(self.mask_by_intersection(shape).any())

    def mask_by_dt(self, dt: Union[datetime, TimeInterval]) -> np.ndarray:
        """
        A boolean array marking the points whose times intersect the given
        datetime or time interval, following TimeInterval.intersects (intervals
        are right-open; instants intersect only an equal instant). Points
        without a time are never marked.

        Args:
            dt:
                A datetime or TimeInterval

        Returns:
            np.ndarray of bool
        """
        if isinstance(dt, datetime):
            dt = TimeInterval(dt, dt)

        if not isinstance(dt, TimeInterval):
            raise ValueError(f'Unexpected dt object: {dt}')

        q_start, q_end = _datetime_to_datetime64(dt.start), _datetime_to_datetime64(dt.end)
        starts, ends = self._start, self._end
        point_is_instant = starts == ends

        if dt.is_instant:
            hit = np.where(point_is_instant, starts == q_start, (starts <= q_start) & (q_start < ends))
        else:
            hit = np.where(
                point_is_instant,
                (q_start <= starts) & (starts < q_end),
                (ends > q_start) & (starts < q_end)
            )

        # NaT compares False everywhere above, but be explicit
        return hit & ~np.isnat(starts)

    def mask_by_intersection(self, shape: BaseShape) -> np.ndarray:
        """
        A boolean array marking the points that intersect a geoshape, which is
        optionally time-bounded. Matches GeoPoint.intersects.

        Args:
            shape:
                A geoshape

        Returns:
            np.ndarray of bool
        """
        if isinstance(shape, GeoPoint):
            # Points intersect only an equal point
            return self._equals_mask(shape)

        spatial = self._spatial_mask(shape)
        if spatial is None:
            # The shape treats points specially; ask it about each one in full
            return np.fromiter((x.intersects(shape) for x in self), dtype=np.bool_, count=len(self))

        if shape.dt is None:
            return spatial

        # A point intersects a time-bounded shape when it lies inside it: its
        # time (if any) must intersect the shape's, and also be contained by it
        has_time = ~np.isnat(self._start)
        return spatial & (~has_time | (self.mask_by_dt(shape.dt) & self._time_contained_mask(shape.dt)))

    def mask_contained_by(self, shape: BaseShape) -> np.ndarray:
        """
        A boolean array marking the points contained by a geoshape, which is
        optionally time-bounded. Matches shape.contains(point): a point's time,
        if it has one, must lie within the shape's time, if it has one.

        Args:
            shape:
                A geoshape

        Returns:
            np.ndarray of bool
        """
        spatial = self._spatial_mask(shape)
        if spatial is None:
            # The shape treats points specially; ask it about each one in full
            return np.fromiter((shape.contains(x) for x in self), dtype=np.bool_, count=len(self))

        if shape.dt is None:
            return spatial

        return spatial & (np.isnat(self._start) | self._time_contained_mask(shape.dt))

    def to_feature_collection(self) -> FeatureCollection:
        """Converts to a FeatureCollection of GeoPoints"""
        shapes: List[BaseShape] = list(self)
        return FeatureCollection(shapes)

    def to_geopandas(self, include_properties: Optional[List[str]] = None):
        """
        Converts to a geopandas GeoDataFrame without creating a Python object
        per point. As with FeatureCollection.to_geopandas, properties become
        columns, including datetime_start/datetime_end when any point has a time.

        Args:
            include_properties: (Optional)
                The property names to include as columns. Defaults to all.

        Returns:
            geopandas.GeoDataFrame
        """
        import_optional('pandas')
        import pandas as pd
        import_optional('geopandas')
        import geopandas as gpd

        # Let pandas infer object columns' types from their values (e.g. ints
        # with gaps become float64 with NaN), as it does for FeatureCollection
        columns: Dict[str, Any] = {
            name: pd.Series(values.tolist()) if values.dtype == object else values
            for name, values in self._props.items()
        }
        if not np.isnat(self._start).all():
            columns[_DT_START_KEY] = pd.to_datetime(self._start).tz_localize('UTC')
            columns[_DT_END_KEY] = pd.to_datetime(self._end).tz_localize('UTC')

        names = include_properties or list(columns)
        return gpd.GeoDataFrame(
            data=pd.DataFrame(
                {name: columns.get(name) for name in names},
                index=pd.RangeIndex(len(self))
            ),
            geometry=gpd.points_from_xy(self._lon, self._lat, self._z),
        )

    def to_points(self) -> List[GeoPoint]:
        """Creates a GeoPoint for every point"""
        return list(self)

    def to_shapely(self) -> np.ndarray:
        """Converts to a numpy array of shapely Points"""
        import_optional('shapely')
        import shapely

        if self._z is None:
            return cast(np.ndarray, shapely.points(self._lon, self._lat))
        return cast(np.ndarray, shapely.points(self._lon, self._lat, self._z))

    def to_track(self) -> Track:
        """Converts to a Track of GeoPoints. Every point must have a time."""
        shapes: List[BaseShape] = list(self)
        return Track(shapes)

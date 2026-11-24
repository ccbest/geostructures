"""
Conversion between shapes and arrays of shapely geometries.

Collections read from shapely or geopandas keep the source geometries and
create each shape only when it is first accessed (see LazyShapes), so filters
and exports that work from arrays never need to create most of them. Shapes
are built from coordinates read directly from the geometries rather than via
WKT text, which GEOS doesn't always write precisely enough to round-trip a
float.
"""

__all__ = ['LazyShapes', 'as_geometry_array', 'shape_from_shapely', 'shapes_to_shapely']

from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, cast

import numpy as np

from geostructures._base import BaseShape
from geostructures.coordinates import Coordinate
from geostructures.multistructures import MultiGeoLineString, MultiGeoPoint, MultiGeoPolygon
from geostructures.structures import (
    GeoBox, GeoCircle, GeoEllipse, GeoLineString, GeoPoint, GeoPolygon, GeoRing, PolygonBase
)
from geostructures.time import GEOTIME_TYPE, _TimeColumns
from geostructures.utils.conditional_imports import import_optional

# shapely.get_type_id values for the geometry types geostructures represents
_POINT, _LINESTRING, _POLYGON, _MULTIPOINT, _MULTILINESTRING, _MULTIPOLYGON = 0, 1, 3, 4, 5, 6
_SUPPORTED_TYPE_IDS = (_POINT, _LINESTRING, _POLYGON, _MULTIPOINT, _MULTILINESTRING, _MULTIPOLYGON)


def _shapely():
    import_optional('shapely')
    import shapely  # pylint: disable=import-outside-toplevel
    return shapely


def _has_m(geometry) -> bool:
    # M support arrived in shapely 2.1
    return bool(getattr(geometry, 'has_m', False))


def as_geometry_array(geometries: Any) -> np.ndarray:
    """
    Converts shapely geometries (a GeometryCollection, GeoSeries, array, or any
    iterable) into a 1-D object array, rejecting anything that can't become a
    shape: unsupported or missing geometries, empty geometries, and NaN or
    infinite coordinates.
    """
    shapely = _shapely()
    if isinstance(geometries, shapely.GeometryCollection):
        geometries = shapely.get_parts(geometries)

    if hasattr(geometries, '__array__'):
        # numpy arrays, and geopandas' GeometryArray/GeoSeries
        arr = np.asarray(geometries, dtype=object)
    else:
        arr = np.empty(len(geometries), dtype=object)
        arr[:] = list(geometries)
    if arr.ndim != 1:
        raise ValueError('Expected a one-dimensional sequence of geometries')

    type_ids = shapely.get_type_id(arr)
    unsupported = ~np.isin(type_ids, _SUPPORTED_TYPE_IDS)
    if unsupported.any():
        bad = arr[np.flatnonzero(unsupported)[0]]
        raise ValueError(
            f'Unrecognized geometry type: {bad.geom_type if bad is not None else None}'
        )

    if shapely.is_empty(arr).any():
        raise ValueError('Empty geometries cannot be converted to shapes')

    if not np.isfinite(shapely.get_coordinates(arr)).all():
        raise ValueError('Geometries may not contain NaN or infinite coordinates')

    return arr


def _coordinates(geometry, has_z: bool, has_m: bool) -> List[Coordinate]:
    kwargs = {'include_m': True} if has_m else {}
    coords = _shapely().get_coordinates(geometry, include_z=has_z, **kwargs).tolist()
    if has_z and has_m:
        return [Coordinate(x, y, z, m) for x, y, z, m in coords]
    if has_z:
        return [Coordinate(x, y, z) for x, y, z in coords]
    if has_m:
        return [Coordinate(x, y, m=m) for x, y, m in coords]
    return [Coordinate(x, y) for x, y in coords]


def _polygon(geometry, has_z: bool, has_m: bool, **kwargs) -> GeoPolygon:
    holes = [GeoPolygon(_coordinates(ring, has_z, has_m)) for ring in geometry.interiors]
    return GeoPolygon(
        _coordinates(geometry.exterior, has_z, has_m),
        holes=holes or None,
        **kwargs
    )


def shape_from_shapely(
    geometry: Any,
    dt: Optional[GEOTIME_TYPE] = None,
    properties: Optional[Dict] = None,
) -> BaseShape:
    """
    Creates the shape corresponding to a shapely geometry. Shapes are assembled
    exactly as the from_wkt constructors assemble them (holes as separate
    polygons, multishapes from untimed parts), so the two only differ where
    WKT text would have lost precision.
    """
    geom_type = geometry.geom_type
    has_z, has_m = geometry.has_z, _has_m(geometry)

    if geom_type == 'Point':
        return GeoPoint(_coordinates(geometry, has_z, has_m)[0], dt=dt, properties=properties)

    if geom_type == 'LineString':
        return GeoLineString(_coordinates(geometry, has_z, has_m), dt=dt, properties=properties)

    if geom_type == 'Polygon':
        return _polygon(geometry, has_z, has_m, dt=dt, properties=properties)

    if geom_type == 'MultiPoint':
        return MultiGeoPoint(
            [GeoPoint(coord) for coord in _coordinates(geometry, has_z, has_m)],
            dt=dt,
            properties=properties
        )

    if geom_type == 'MultiLineString':
        return MultiGeoLineString(
            [GeoLineString(_coordinates(part, has_z, has_m)) for part in geometry.geoms],
            dt=dt,
            properties=properties
        )

    if geom_type == 'MultiPolygon':
        return MultiGeoPolygon(
            [_polygon(part, has_z, has_m) for part in geometry.geoms],
            dt=dt,
            properties=properties
        )

    raise ValueError(f'Unrecognized geometry type: {geom_type}')


def _in_coordinate_range(coords: np.ndarray) -> np.ndarray:
    """Rows of (lon, lat) that Coordinate would store unchanged"""
    return (
        (-180 <= coords[:, 0]) & (coords[:, 0] < 180)
        & (-90 <= coords[:, 1]) & (coords[:, 1] <= 90)
    )


class _Source:
    """The data a LazyShapes (and every subset taken from it) creates its shapes from"""

    __slots__ = ('geometries', 'times', 'properties', 'shapes')

    def __init__(
        self, geometries: np.ndarray, times: _TimeColumns, properties: List, shapes: List
    ):
        self.geometries = geometries
        self.times = times
        self.properties = properties
        self.shapes = shapes


class LazyShapes(Sequence[BaseShape]):
    """
    A sequence of shapes created on demand from shapely geometries. Each shape
    is created the first time it is accessed and then kept, and subsets taken
    with take() share those shapes, so a given row is always the same object.

    Args:
        geometries:
            An object array of supported, non-empty shapely geometries (see
            as_geometry_array)

        times: (Optional)
            The time each shape is constructed with

        properties: (Optional)
            The properties to construct each shape with
    """

    def __init__(
        self,
        geometries: np.ndarray,
        times: Optional[_TimeColumns] = None,
        properties: Optional[Sequence[Optional[Dict]]] = None,
    ):
        n = len(geometries)
        self._source = _Source(
            geometries,
            times if times is not None else _TimeColumns.from_intervals([None] * n),
            list(properties) if properties is not None else [None] * n,
            [None] * n,
        )
        self._rows = np.arange(n)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, item):  # type: ignore[override]
        if isinstance(item, slice):
            return self.take(np.arange(len(self._rows))[item])
        return self._shape(int(self._rows[item]))

    def __iter__(self) -> Iterator[BaseShape]:
        for row in self._rows.tolist():
            yield self._shape(row)

    def _shape(self, row: int) -> BaseShape:
        source = self._source
        shape = source.shapes[row]
        if shape is None:
            shape = shape_from_shapely(
                source.geometries[row], source.times.interval(row), source.properties[row]
            )
            source.shapes[row] = shape
        return shape

    def take(self, idx: np.ndarray) -> 'LazyShapes':
        """A subset of these shapes, sharing any already created"""
        new = LazyShapes.__new__(LazyShapes)
        new._source = self._source
        new._rows = self._rows[idx]
        return new

    @property
    def geometries(self) -> np.ndarray:
        """The source geometries"""
        return self._source.geometries[self._rows]

    def time_columns(self) -> _TimeColumns:
        """The time each shape is (or will be) constructed with"""
        return self._source.times.take(self._rows)

    def created(self) -> np.ndarray:
        """Whether each shape has been created yet"""
        shapes = self._source.shapes
        return np.array([shapes[row] is not None for row in self._rows.tolist()], dtype=bool)

    def properties_at(self, i: int) -> Dict:
        """
        What shape.properties would return for row i, without creating the
        shape (see BaseShape.properties)
        """
        row = int(self._rows[i])
        props = dict(self._source.properties[row] or {})
        dt = self._source.times.interval(row)
        if dt is not None:
            props['datetime_start'], props['datetime_end'] = dt.start, dt.end
        return props

    def point_columns(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Whether each row is a GeoPoint, and (N, 2) arrays of their longitudes and
        latitudes (NaN for other rows), read from the geometries where they
        already match what the shape would store.
        """
        shapely = _shapely()
        geometries = self.geometries
        is_point = shapely.get_type_id(geometries) == _POINT
        coords = np.full((len(geometries), 2), np.nan)

        point_idx = np.flatnonzero(is_point)
        if len(point_idx):
            coords[point_idx] = shapely.get_coordinates(geometries[point_idx])
            # Coordinate wraps values outside its range; read those from the shape
            for i in point_idx[~_in_coordinate_range(coords[point_idx])]:
                coordinate = self[int(i)].coordinate  # type: ignore[attr-defined]
                coords[i] = coordinate.longitude, coordinate.latitude

        return is_point, coords

    def unchanged_geometries(self) -> np.ndarray:
        """
        Rows whose source geometry is exactly what shape.to_shapely() would
        return: points and linestrings (whose coordinates the shape keeps in
        order) within Coordinate's range and without M values. Polygons are
        excluded because shapes normalize their ring orientation.
        """
        shapely = _shapely()
        geometries = self.geometries
        candidates = np.isin(shapely.get_type_id(geometries), (_POINT, _LINESTRING))
        if hasattr(shapely, 'has_m'):
            candidates &= ~shapely.has_m(geometries)

        idx = np.flatnonzero(candidates)
        if len(idx):
            coords, coord_geom = shapely.get_coordinates(geometries[idx], return_index=True)
            out_of_range = ~_in_coordinate_range(coords)
            candidates[idx[np.unique(coord_geom[out_of_range])]] = False

        return candidates


def _positions(coords: Sequence[Coordinate]) -> Optional[Tuple[List[List[float]], int]]:
    """Coordinate positions and their shared dimension, or None if mixed 2D/3D"""
    positions = [coord.to_position() for coord in coords]
    dims = {len(x) for x in positions}
    if len(dims) != 1:
        return None
    return positions, dims.pop()


def _ring_positions(rings) -> Optional[Tuple[List[List[List[float]]], int]]:
    """Positions for closed rings sharing one dimension, or None"""
    converted, dims = [], set()
    for ring in rings:
        result = _positions(ring)
        if result is None or len(ring) < 4 or ring[0] != ring[-1]:
            # Mixed dimensions, or a ring shapely's Polygon constructor would
            # have to repair; left to the shape's own conversion
            return None
        converted.append(result[0])
        dims.add(result[1])
    if len(dims) != 1:
        return None
    return converted, dims.pop()


class _RaggedBuilder:
    """Accumulates one geometry type and dimension in shapely's ragged array layout"""

    def __init__(self, geometry_type, levels: int):
        self.geometry_type = geometry_type
        self.rows: List[int] = []
        self.coords: List[List[float]] = []
        # Offsets from innermost (coordinates) to outermost (geometries)
        self.offsets: List[List[int]] = [[0] for _ in range(levels)]

    def build(self) -> np.ndarray:
        offsets = tuple(np.array(x, dtype=np.int64) for x in self.offsets) or None
        return _shapely().from_ragged_array(
            self.geometry_type, np.array(self.coords, dtype=np.float64), offsets
        )


# Shapes converted with PolygonBase._to_shapely. Exact types, so that a subclass
# overriding _to_shapely is converted with its own implementation
_POLYGON_TYPES = (GeoPolygon, GeoBox, GeoCircle, GeoEllipse, GeoRing)


def shapes_to_shapely(shapes: Sequence[BaseShape]) -> np.ndarray:
    """
    Converts shapes into an object array of shapely geometries, each identical
    to shape.to_shapely(), building each geometry type in a single
    vectorized call. Results are stored as each shape's to_shapely() cache.
    """
    shapely = _shapely()
    gtype = shapely.GeometryType
    out = np.empty(len(shapes), dtype=object)

    rows: Sequence[int] = range(len(shapes))
    if isinstance(shapes, LazyShapes):
        unchanged = shapes.unchanged_geometries()
        out[unchanged] = shapes.geometries[unchanged]
        rows = np.flatnonzero(~unchanged).tolist()

    builders: Dict[Tuple[Any, int], _RaggedBuilder] = {}

    def builder(geometry_type, dim: int, levels: int) -> _RaggedBuilder:
        key = (geometry_type, dim)
        if key not in builders:
            builders[key] = _RaggedBuilder(geometry_type, levels)
        return builders[key]

    fallback: List[int] = []
    for row in rows:
        shape = shapes[row]
        if shape.__dict__.get('_shapely') is not None:
            out[row] = shape.__dict__['_shapely']
            continue

        if type(shape) is GeoPoint:  # pylint: disable=unidiomatic-typecheck
            position = shape.coordinate.to_position()
            target = builder(gtype.POINT, len(position), 0)
            target.rows.append(row)
            target.coords.append(position)
            continue

        if type(shape) is GeoLineString:  # pylint: disable=unidiomatic-typecheck
            result = _positions(shape.vertices)
            if result is not None:
                target = builder(gtype.LINESTRING, result[1], 1)
                target.rows.append(row)
                target.coords += result[0]
                target.offsets[0].append(len(target.coords))
                continue

        elif type(shape) is MultiGeoPoint:  # pylint: disable=unidiomatic-typecheck
            result = _positions([x.centroid for x in shape.geoshapes])
            if result is not None:
                target = builder(gtype.MULTIPOINT, result[1], 1)
                target.rows.append(row)
                target.coords += result[0]
                target.offsets[0].append(len(target.coords))
                continue

        elif type(shape) is MultiGeoLineString:  # pylint: disable=unidiomatic-typecheck
            parts = [_positions(x.vertices) for x in shape.geoshapes]
            dims = {x[1] for x in parts if x is not None}
            if None not in parts and len(dims) == 1:
                target = builder(gtype.MULTILINESTRING, dims.pop(), 2)
                target.rows.append(row)
                for part in parts:
                    target.coords += part[0]  # type: ignore[index]
                    target.offsets[0].append(len(target.coords))
                target.offsets[1].append(len(target.offsets[0]) - 1)
                continue

        elif type(shape) is MultiGeoPolygon:  # pylint: disable=unidiomatic-typecheck
            polygons = [_ring_positions(rings) for rings in shape.linear_rings()]
            dims = {x[1] for x in polygons if x is not None}
            if None not in polygons and len(dims) == 1:
                target = builder(gtype.MULTIPOLYGON, dims.pop(), 3)
                target.rows.append(row)
                for polygon in polygons:
                    for ring in polygon[0]:  # type: ignore[index]
                        target.coords += ring
                        target.offsets[0].append(len(target.coords))
                    target.offsets[1].append(len(target.offsets[0]) - 1)
                target.offsets[2].append(len(target.offsets[1]) - 1)
                continue

        elif type(shape) in _POLYGON_TYPES:
            rings = _ring_positions(cast(PolygonBase, shape).linear_rings())
            if rings is not None:
                target = builder(gtype.POLYGON, rings[1], 2)
                target.rows.append(row)
                for ring in rings[0]:
                    target.coords += ring
                    target.offsets[0].append(len(target.coords))
                target.offsets[1].append(len(target.offsets[0]) - 1)
                continue

        fallback.append(row)

    for target in builders.values():
        out[target.rows] = target.build()

    for row in fallback:
        out[row] = shapes[row].to_shapely()

    # Keep each shape's to_shapely() consistent with what was returned here
    for row in rows:
        shape = shapes[row]
        if shape.__dict__.get('_shapely') is None:
            shape.__dict__['_shapely'] = out[row]

    return out


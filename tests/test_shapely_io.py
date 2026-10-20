"""
Collections read from shapely/geopandas create their shapes on demand, and
exports build shapely geometries from arrays. These tests hold both to the
behavior of the previous implementations, which went through WKT text one
shape at a time (reproduced here as references).
"""
from datetime import datetime, timedelta, timezone
import random

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely

from geostructures import (
    Coordinate, FeatureCollection, GeoBox, GeoCircle, GeoEllipse, GeoLineString, GeoPoint,
    GeoPolygon, GeoRing, MultiGeoLineString, MultiGeoPoint, MultiGeoPolygon, Track
)
from geostructures._columnar import ShapeColumns
from geostructures._shapely_io import LazyShapes, shape_from_shapely, shapes_to_shapely
from geostructures.time import TimeInterval
from tests.test_columnar import _random_shapes


_CONV_MAP = {
    'Point': GeoPoint,
    'LineString': GeoLineString,
    'Polygon': GeoPolygon,
    'MultiPoint': MultiGeoPoint,
    'MultiLineString': MultiGeoLineString,
    'MultiPolygon': MultiGeoPolygon,
}


def _signature(shape):
    """Everything that identifies a shape: type, exact coordinates and structure, time, properties"""
    def coords(seq):
        return tuple((c.longitude, c.latitude, c.z, c.m) for c in seq)

    if isinstance(shape, GeoPoint):
        geometry = coords([shape.coordinate])
    elif isinstance(shape, GeoLineString):
        geometry = coords(shape.vertices)
    elif isinstance(shape, GeoPolygon):
        geometry = (coords(shape.outline), tuple(_signature(h) for h in shape.holes))
    else:
        geometry = tuple(_signature(x) for x in shape.geoshapes)
    return type(shape).__name__, geometry, shape.dt, repr(shape._properties)


def _old_from_geopandas(df, time_start_field='datetime_start', time_end_field='datetime_end'):
    """
    The previous FeatureCollection.from_geopandas, shape by shape via WKT - with
    its missing-time handling fixed (NaT is truthy, so a start without an end
    used to become a TimeInterval ending at NaT rather than an instant)
    """
    def _get_dt(rec):
        dt_start, dt_end = rec.get(time_start_field), rec.get(time_end_field)
        dt_start = dt_start if isinstance(dt_start, datetime) and not pd.isnull(dt_start) else None
        dt_end = dt_end if isinstance(dt_end, datetime) and not pd.isnull(dt_end) else None
        if dt_start is None and dt_end is None:
            return None
        if dt_start is None or dt_end is None or dt_start == dt_end:
            return dt_start or dt_end
        return TimeInterval(dt_start, dt_end)

    prop_fields = [x for x in df.columns if x not in (time_start_field, time_end_field, 'geometry')]
    shapes = []
    for record in df.to_dict('records'):
        props = {k: v for k, v in record.items() if k in prop_fields}
        shapes.append(_CONV_MAP[record['geometry'].geom_type].from_wkt(
            record['geometry'].wkt, dt=_get_dt(record), properties=props
        ))
    return shapes


def _old_to_geopandas(shapes):
    """The previous CollectionBase.to_geopandas"""
    keys = set(_key for x in shapes for _key in x.properties.keys())
    return gpd.GeoDataFrame(
        data=pd.DataFrame([{key: x.properties.get(key) for key in keys} for x in shapes]),
        geometry=gpd.GeoSeries.from_wkt([x.to_wkt() for x in shapes])
    )


def _nice(rnd, low, high):
    # Multiples of 1/8 print exactly in WKT, so the old WKT path is exact too
    return rnd.randint(int(low * 8), int(high * 8)) / 8


def _random_geometries(rnd, n):
    """Mixed geometries, including Z/M values, holes, clockwise rings and out-of-range coordinates"""
    def coord(dims):
        lon = _nice(rnd, -200, 200) if rnd.random() < 0.1 else _nice(rnd, -20, 20)
        lat = _nice(rnd, -20, 20)
        return (lon, lat, *(_nice(rnd, 0, 10) for _ in range(dims - 2)))

    def ring(dims, cx, cy, size, clockwise):
        pts = [(cx - size, cy - size), (cx + size, cy - size), (cx + size, cy + size), (cx - size, cy + size)]
        if clockwise:
            pts = pts[::-1]
        pts = [(*p, *(_nice(rnd, 0, 10) for _ in range(dims - 2))) for p in pts]
        return [*pts, pts[0]]

    def polygon(dims):
        cx, cy = _nice(rnd, -20, 20), _nice(rnd, -20, 20)
        holes = [ring(dims, cx, cy, 0.5, rnd.random() < 0.5)] if rnd.random() < 0.4 else []
        return shapely.Polygon(ring(dims, cx, cy, 2, rnd.random() < 0.5), holes)

    geometries = []
    for _ in range(n):
        dims = rnd.choice([2, 2, 2, 3])
        kind = rnd.random()
        if kind < 0.4:
            geom = shapely.Point(coord(dims))
        elif kind < 0.55:
            geom = shapely.LineString([coord(dims) for _ in range(rnd.randint(2, 5))])
        elif kind < 0.75:
            geom = polygon(dims)
        elif kind < 0.85:
            geom = shapely.MultiPoint([coord(dims) for _ in range(rnd.randint(1, 4))])
        elif kind < 0.92:
            geom = shapely.MultiLineString([[coord(dims) for _ in range(3)] for _ in range(2)])
        else:
            geom = shapely.MultiPolygon([polygon(dims) for _ in range(2)])
        geometries.append(geom)

    # M values, via WKT
    geometries += [
        shapely.from_wkt('POINT M (1 2 3)'),
        shapely.from_wkt('POINT ZM (1 2 3 4)'),
        shapely.from_wkt('LINESTRING M (0 0 1, 1 1 2)'),
        shapely.from_wkt('POLYGON M ((0 0 1, 1 0 2, 1 1 3, 0 0 1))'),
    ]
    return geometries


_T0 = datetime(2020, 1, 1, tzinfo=timezone.utc)


def _random_dataframe(rnd, n):
    geometries = _random_geometries(rnd, n)
    starts, ends = [], []
    for _ in geometries:
        roll = rnd.random()
        start = _T0 + timedelta(hours=rnd.randint(0, 5))
        if roll < 0.2:
            starts.append(None)
            ends.append(None)
        elif roll < 0.4:
            starts.append(start)
            ends.append(None)
        elif roll < 0.5:
            starts.append(start)
            ends.append(start)
        else:
            starts.append(start)
            ends.append(start + timedelta(hours=rnd.randint(1, 3)))
    return gpd.GeoDataFrame(
        {
            'speed': [rnd.random() for _ in geometries],
            'label': [rnd.choice(['a', 'b', None]) for _ in geometries],
            'count': [rnd.randint(0, 9) for _ in geometries],
            'datetime_start': pd.to_datetime(starts, utc=True),
            'datetime_end': pd.to_datetime(ends, utc=True),
        },
        geometry=geometries,
    )


@pytest.mark.parametrize('seed', range(3))
def test_shape_from_shapely_matches_wkt_path(seed):
    rnd = random.Random(seed)
    for geom in _random_geometries(rnd, 200):
        expected = _CONV_MAP[geom.geom_type].from_wkt(geom.wkt)
        assert _signature(shape_from_shapely(geom)) == _signature(expected), geom.wkt


def test_shape_from_shapely_is_exact():
    # GEOS doesn't always write enough digits to round-trip a float through WKT
    xy = np.random.default_rng(0).uniform(-1, 1, (2_000, 2))
    for (x, y), geom in zip(xy.tolist(), shapely.points(xy)):
        coord = shape_from_shapely(geom).coordinate
        assert (coord.longitude, coord.latitude) == (x, y)


@pytest.mark.parametrize('seed', range(3))
def test_from_geopandas_matches_previous_implementation(seed):
    df = _random_dataframe(random.Random(seed), 200)
    expected = [_signature(x) for x in _old_from_geopandas(df)]

    fcol = FeatureCollection.from_geopandas(df)
    assert isinstance(fcol._shapes, LazyShapes)
    assert [_signature(x) for x in fcol] == expected
    # The same row is always the same object
    assert all(x is y for x, y in zip(fcol, fcol))

    timed = df[df['datetime_start'].notna()]
    expected_track = sorted(_old_from_geopandas(timed), key=lambda x: x.start)
    assert [_signature(x) for x in Track.from_geopandas(timed)] == \
        [_signature(x) for x in expected_track]

    with pytest.raises(ValueError, match='must have an associated time value'):
        Track.from_geopandas(df)


@pytest.mark.parametrize('seed', range(3))
def test_lazy_columns_match_shape_columns(seed):
    df = _random_dataframe(random.Random(seed), 300)
    lazy = ShapeColumns(FeatureCollection.from_geopandas(df)._shapes)
    eager = ShapeColumns(_old_from_geopandas(df))

    np.testing.assert_array_equal(lazy.is_point, eager.is_point)
    np.testing.assert_array_equal(lazy.point_coords, eager.point_coords)
    np.testing.assert_array_equal(lazy.envelopes, eager.envelopes)
    np.testing.assert_array_equal(lazy.has_dt, eager.has_dt)
    for lazy_times, eager_times in zip(lazy._times[:2], eager._times[:2]):
        np.testing.assert_array_equal(lazy_times, eager_times)


@pytest.mark.parametrize('seed', range(3))
def test_lazy_filters_match_eager(seed):
    rnd = random.Random(seed)
    df = _random_dataframe(rnd, 300)
    lazy = FeatureCollection.from_geopandas(df)
    eager = FeatureCollection(_old_from_geopandas(df))
    lazy_rows = {id(x): i for i, x in enumerate(lazy)}
    eager_rows = {id(x): i for i, x in enumerate(eager)}

    queries = [
        GeoBox(Coordinate(-10., 10.), Coordinate(10., -10.)),
        GeoPolygon([Coordinate(0., 0.), Coordinate(15., 0.), Coordinate(15., 15.), Coordinate(0., 0.)]),
        GeoCircle(Coordinate(0., 0.), 800_000, dt=TimeInterval(_T0, _T0 + timedelta(hours=3))),
    ]
    for query in queries:
        for method in ('filter_by_intersection', 'filter_contained_by', 'filter_contains'):
            assert [lazy_rows[id(x)] for x in getattr(lazy, method)(query)] == \
                [eager_rows[id(x)] for x in getattr(eager, method)(query)], (method, query)
        assert lazy.intersects(query) == eager.intersects(query)

    window = TimeInterval(_T0 + timedelta(hours=1), _T0 + timedelta(hours=3))
    assert [lazy_rows[id(x)] for x in lazy.filter_by_dt(window)] == \
        [eager_rows[id(x)] for x in eager.filter_by_dt(window)]


def test_point_workflows_create_few_shapes():
    n = 5_000
    xy = np.random.default_rng(0).uniform(-10, 10, (n, 2))
    df = gpd.GeoDataFrame(
        {'datetime_start': pd.date_range('2020-01-01', periods=n, freq='s', tz='UTC')},
        geometry=shapely.points(xy),
    )
    track = Track.from_geopandas(df)
    filtered = track.filter_by_intersection(GeoBox(Coordinate(0., 1.), Coordinate(1., 0.)))
    filtered = filtered[datetime(2020, 1, 1):datetime(2020, 1, 1, 1)]
    out = filtered.to_geopandas()

    assert len(out) == len(filtered) > 0
    # Nothing above needed a shape object
    assert not track._shapes.created().any()

    # Indexing creates just that shape
    _ = filtered[0]
    assert track._shapes.created().sum() == 1


@pytest.mark.parametrize('seed', range(3))
def test_to_shapely_matches_shapes(seed):
    shapes, _ = _random_shapes(random.Random(seed), 300)
    shapes += [
        GeoPoint(Coordinate(1., 2., z=3.)),
        GeoLineString([Coordinate(0., 0., z=1.), Coordinate(1., 1., z=2.)]),
        GeoPolygon([
            Coordinate(0., 0., z=1.), Coordinate(1., 0., z=1.),
            Coordinate(1., 1., z=1.), Coordinate(0., 0., z=1.)
        ]),
        GeoRing(Coordinate(0., 0.), 1_000, 2_000),
        GeoRing(Coordinate(0., 0.), 1_000, 2_000, angle_min=0, angle_max=90),
        GeoEllipse(Coordinate(0., 0.), 2_000, 1_000, 45),
        GeoPoint(Coordinate(1., 2., m=3.)),
    ]
    # Compare against fresh copies, since conversion caches on the shapes
    expected = [shapely.to_wkb(x.copy().to_shapely()) for x in shapes]

    converted = FeatureCollection(shapes).to_shapely()
    assert [shapely.to_wkb(x) for x in converted] == expected
    # And each shape's own to_shapely() now returns the same geometry
    assert all(x.to_shapely() is g for x, g in zip(shapes, converted))


@pytest.mark.parametrize('seed', range(3))
def test_lazy_to_shapely_matches_shapes(seed):
    df = _random_dataframe(random.Random(seed), 300)
    expected = [shapely.to_wkb(x.to_shapely()) for x in _old_from_geopandas(df)]
    assert [shapely.to_wkb(x) for x in FeatureCollection.from_geopandas(df).to_shapely()] == expected


@pytest.mark.parametrize('seed', range(3))
def test_to_geopandas_matches_previous_implementation(seed):
    rnd = random.Random(seed)
    df = _random_dataframe(rnd, 200)
    shapes, _ = _random_shapes(rnd, 100)
    shapes += [GeoPoint(Coordinate(1., 2., m=3.)), GeoLineString([Coordinate(0., 0., z=1., m=2.), Coordinate(1., 1., z=1., m=2.)])]

    for collection, reference in (
        (FeatureCollection.from_geopandas(df), _old_from_geopandas(df)),
        (FeatureCollection([x.copy() for x in shapes]), shapes),
    ):
        actual, expected = collection.to_geopandas(), _old_to_geopandas(reference)
        assert list(actual.columns) == list(expected.columns)
        pd.testing.assert_frame_equal(
            pd.DataFrame(actual.drop(columns='geometry')),
            pd.DataFrame(expected.drop(columns='geometry')),
        )
        # The previous implementation went via to_wkt(), which rounds to 12
        # decimal places; geometries now carry the shapes' exact coordinates.
        # Full GeoRings are excluded: their to_wkt() and to_shapely() draw
        # different rings, and geometries now consistently follow to_shapely()
        # (checked in test_to_shapely_matches_shapes)
        comparable = np.array([
            not (isinstance(x, GeoRing) and x.angle_min == 0 and x.angle_max == 360)
            for x in reference
        ])
        assert list(actual.geometry.geom_type) == list(expected.geometry.geom_type)
        assert list(actual.geometry.has_z) == list(expected.geometry.has_z)
        assert shapely.equals_exact(
            actual.geometry.values[comparable], expected.geometry.values[comparable],
            tolerance=1e-11
        ).all()


def test_from_geopandas_geometry_only():
    df = gpd.GeoDataFrame(geometry=[shapely.Point(0, 1), shapely.Point(1, 2)])
    assert list(FeatureCollection.from_geopandas(df)) == \
        [GeoPoint(Coordinate(0., 1.)), GeoPoint(Coordinate(1., 2.))]


def test_from_shapely_inputs():
    geometries = [shapely.Point(0, 1), shapely.LineString([(0, 0), (1, 1)])]
    expected = [GeoPoint(Coordinate(0., 1.)), GeoLineString([Coordinate(0., 0.), Coordinate(1., 1.)])]

    for source in (
        shapely.GeometryCollection(geometries),
        geometries,
        np.array(geometries, dtype=object),
        gpd.GeoSeries(geometries),
    ):
        assert list(FeatureCollection.from_shapely(source)) == expected


def test_from_shapely_rejects_unconvertible():
    for bad in (
        [None],
        [shapely.Point()],
        [shapely.GeometryCollection([shapely.Point(0, 0)])],
        [shapely.LinearRing([(0, 0), (1, 0), (1, 1), (0, 0)])],
        [shapely.Point(float('nan'), 0)],
    ):
        with pytest.raises(ValueError):
            FeatureCollection.from_shapely(bad)


def test_ragged_array_round_trip():
    for shapes in (
        [GeoPoint(Coordinate(float(i), 1.)) for i in range(5)],
        [GeoPolygon([
            Coordinate(i, 0.), Coordinate(i + 1., 0.), Coordinate(i + 1., 1.), Coordinate(i, 0.)
        ]) for i in range(3)],
    ):
        fcol = FeatureCollection(shapes)
        geometry_type, coords, offsets = fcol.to_ragged_array()
        assert list(FeatureCollection.from_ragged_array(geometry_type, coords, offsets)) == shapes

    with pytest.raises(ValueError):
        FeatureCollection([GeoPoint(Coordinate(0., 0.)), GeoLineString([Coordinate(0., 0.), Coordinate(1., 1.)])]).to_ragged_array()


def test_time_filters_floor_nanoseconds():
    # pandas Timestamps carry nanoseconds, which TimeInterval floors to microseconds
    base = pd.Timestamp('2020-01-01', tz='UTC')
    times = [base + pd.Timedelta(nanoseconds=n) for n in (0, 1, 500, 999, 1_000, 1_001)]
    shapes = [GeoPoint(Coordinate(0., 0.), dt=t) for t in times]
    shapes += [GeoPoint(Coordinate(0., 0.), dt=TimeInterval(a, b)) for a in times for b in times if a < b]
    fcol = FeatureCollection(shapes)

    for start in times:
        for end in times:
            if end < start:
                continue
            query = TimeInterval(start, end)
            assert [id(x) for x in fcol.filter_by_dt(query)] == \
                [id(x) for x in shapes if query.intersects(x.dt)]

    assert GeoPoint(Coordinate(0., 0.), dt=times[3]).dt == GeoPoint(Coordinate(0., 0.), dt=base).dt
    assert GeoPoint(Coordinate(0., 0.), dt=times[3]).dt != GeoPoint(Coordinate(0., 0.), dt=times[4]).dt

    track = Track(shapes)
    for t in times:
        t_floor = t.floor('us')
        assert [id(x) for x in track[t:]] == [id(x) for x in track if t_floor <= x.start]
        assert [id(x) for x in track[:t]] == [id(x) for x in track if x.end < t_floor]


def test_lazy_shapes_take_shares_created_shapes():
    lazy = LazyShapes(np.array([shapely.Point(0, 0), shapely.Point(1, 1)], dtype=object))
    subset = lazy.take(np.array([1]))
    assert subset[0] is lazy[1]
    assert lazy.created().tolist() == [False, True]
    assert lazy[-1] is lazy[1]
    assert list(lazy[1:]) == [lazy[1]]


def test_shapes_to_shapely_empty():
    assert len(shapes_to_shapely([])) == 0
    assert len(FeatureCollection([]).to_geopandas()) == 0


def test_from_geopandas_time_columns():
    from zoneinfo import ZoneInfo

    paris = ZoneInfo('Europe/Paris')
    points = [shapely.Point(0, 0)] * 4
    starts = pd.Series(pd.to_datetime(['2020-01-01', '2020-01-02', None, None])).dt.tz_localize(paris)
    ends = pd.Series(pd.to_datetime(['2020-01-03', None, '2020-01-04', None]))   # naive: UTC
    df = gpd.GeoDataFrame({'datetime_start': starts, 'datetime_end': ends, 'x': range(4)}, geometry=points)
    fcol = FeatureCollection.from_geopandas(df)

    assert fcol[0].dt == TimeInterval(datetime(2020, 1, 1, tzinfo=paris), datetime(2020, 1, 3))
    assert fcol[0].start.tzinfo is paris and fcol[0].end.tzinfo is timezone.utc
    # A single endpoint becomes an instant, in that endpoint's timezone
    assert fcol[1].dt == TimeInterval(datetime(2020, 1, 2, tzinfo=paris), datetime(2020, 1, 2, tzinfo=paris))
    assert fcol[1].end.tzinfo is paris
    assert fcol[2].dt == TimeInterval(datetime(2020, 1, 4), datetime(2020, 1, 4))
    assert fcol[3].dt is None
    assert [x.properties.get('x') for x in fcol] == [0, 1, 2, 3]

    # Object columns of python datetimes (and non-datetimes, which are ignored)
    df = gpd.GeoDataFrame(
        {'datetime_start': [datetime(2020, 1, 1, tzinfo=paris), datetime(2020, 1, 1), 'n/a', None]},
        geometry=points
    )
    fcol = FeatureCollection.from_geopandas(df)
    assert fcol[0].start == datetime(2020, 1, 1, tzinfo=paris) and fcol[0].start.tzinfo is paris
    assert fcol[1].start == datetime(2020, 1, 1, tzinfo=timezone.utc)
    assert fcol[2].dt is None and fcol[3].dt is None
    assert FeatureCollection.from_geopandas(df).filter_by_dt(datetime(2020, 1, 1)).geoshapes == (fcol[1],)

    df = gpd.GeoDataFrame(
        {'datetime_start': pd.to_datetime(['2020-01-02']), 'datetime_end': pd.to_datetime(['2020-01-01'])},
        geometry=points[:1]
    )
    with pytest.raises(ValueError):
        FeatureCollection.from_geopandas(df)


def test_lazy_collection_mutation():
    full = _random_dataframe(random.Random(3), 200)
    extra = GeoPoint(Coordinate(1., 1.), dt=_T0)
    window = TimeInterval(_T0, _T0 + timedelta(hours=3))
    area = GeoBox(Coordinate(-20., 20.), Coordinate(20., -20.))

    for cls in (FeatureCollection, Track):
        df = full
        if cls is Track:
            df = full[full['datetime_start'].notna() | full['datetime_end'].notna()]
        expected = list(_old_from_geopandas(df))
        col = cls.from_geopandas(df)
        _ = col._columns.envelopes, col._columns.has_dt
        reference = cls(expected)
        col.append(extra)
        reference.append(extra)
        del col[5]
        del reference[5]
        assert len(col) == len(reference)
        assert [x.to_wkt() for x in col] == [x.to_wkt() for x in reference]
        assert [x.dt for x in col] == [x.dt for x in reference]
        assert [x.to_wkt() for x in col.filter_by_intersection(area)] == \
            [x.to_wkt() for x in reference.filter_by_intersection(area)]
        assert [x.dt for x in col.filter_by_dt(window)] == [x.dt for x in reference.filter_by_dt(window)]

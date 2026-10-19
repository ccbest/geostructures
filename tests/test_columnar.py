from datetime import datetime, timedelta, timezone
import itertools
import math

import numpy as np
import pytest
import shapely

from geostructures import (
    Coordinate, FeatureCollection, GeoBox, GeoCircle, GeoEllipse, GeoLineString, GeoPoint,
    GeoPolygon, GeoRing, MultiGeoPolygon, PointCollection, Track
)
from geostructures.columnar import _arrays_equal, _as_column, _as_float_array, _as_time_array
from geostructures.geodesic import set_geodesic_algorithm
from geostructures.time import TimeInterval


T0 = datetime(2020, 1, 1, tzinfo=timezone.utc)
WINDOW = TimeInterval(T0 + timedelta(hours=2), T0 + timedelta(hours=6))


def _make_points():
    """
    Points on a half-degree grid (so plenty sit exactly on shape edges and
    vertices) plus random ones, with a mix of instants, intervals and no time,
    and a property some points don't carry.
    """
    rng = np.random.default_rng(42)
    coords = [Coordinate(x, y) for x, y in itertools.product(np.arange(-3, 3.5, 0.5), repeat=2)]
    coords += [Coordinate(x, y) for x, y in rng.uniform(-3, 3, (200, 2))]
    # Around the antimeridian
    coords += [
        Coordinate(x, y)
        for x, y in itertools.product([178.5, 179, 179.5, -180, -179.5, -179], [-1, 0, 0.5])
    ]

    points = []
    for i, coord in enumerate(coords):
        hour = T0 + timedelta(hours=i % 9)
        if i % 4 == 0:
            dt = None
        elif i % 4 == 1:
            dt = hour
        elif i % 4 == 2:
            dt = TimeInterval(hour, hour + timedelta(hours=1 + i % 3))
        else:
            # A non-UTC timezone, which must compare as the same instant
            dt = hour.astimezone(timezone(timedelta(hours=5)))
        props = {'idx': i, 'parity': 'even' if i % 2 == 0 else 'odd'}
        if i % 3 == 0:
            props['sometimes'] = i
        points.append(GeoPoint(coord, dt=dt, properties=props))

    return points


POINTS = _make_points()

_POLYGON = GeoPolygon([
    Coordinate(-2, -2), Coordinate(2, -2.5), Coordinate(2.5, 1), Coordinate(0, 0),
    Coordinate(0.5, 2), Coordinate(-2, 2), Coordinate(-2, -2)
])
_HOLE = GeoPolygon([Coordinate(-1, -1), Coordinate(1, -1), Coordinate(0, -0.5), Coordinate(-1, -1)])
_ANTIMERIDIAN = GeoPolygon([
    Coordinate(179, -0.5), Coordinate(-179.5, -0.5), Coordinate(-179.5, 1),
    Coordinate(179, 1), Coordinate(179, -0.5)
])

SHAPES = {
    'box': GeoBox(Coordinate(-2, 2), Coordinate(1.5, -1)),
    'box_hole': GeoBox(
        Coordinate(-2, 2), Coordinate(2, -2), holes=[GeoBox(Coordinate(-1, 1), Coordinate(1, -1))]
    ),
    'circle': GeoCircle(Coordinate(0.5, 0.5), 150_000),
    'circle_hole': GeoCircle(Coordinate(0, 0), 250_000, holes=[GeoCircle(Coordinate(0, 0), 60_000)]),
    'polygon': _POLYGON,
    'polygon_hole': GeoPolygon(list(_POLYGON.outline), holes=[_HOLE]),
    'antimeridian': _ANTIMERIDIAN,
    'multipolygon': MultiGeoPolygon([_HOLE, GeoBox(Coordinate(1, 3), Coordinate(3, 1)), _ANTIMERIDIAN]),
    # No vectorized kernel; exercises the per-point fallback
    'ellipse': GeoEllipse(Coordinate(0, 0), 200_000, 100_000, 30),
    'ring': GeoRing(Coordinate(0, 0), 50_000, 200_000, 0, 120),
    'box_ellipse_hole': GeoBox(
        Coordinate(-2, 2), Coordinate(2, -2), holes=[GeoEllipse(Coordinate(0, 0), 50_000, 20_000, 0)]
    ),
    'multipolygon_ellipse': MultiGeoPolygon([_HOLE, GeoEllipse(Coordinate(2, 2), 50_000, 20_000, 0)]),
    'linestring': GeoLineString([Coordinate(-2, -2), Coordinate(2, 2)]),
    'point': POINTS[5],
    'point_untimed': GeoPoint(Coordinate(0.5, 0.5)),
}


def _with_dt(shape, dt):
    return shape.copy().set_dt(dt, inplace=False)


def _nan_safe(props):
    # NaN never equals itself, so swap it for something that does
    return {k: 'NaN' if isinstance(v, float) and math.isnan(v) else v for k, v in props.items()}


def _assert_same_points(actual, expected):
    actual, expected = list(actual), list(expected)
    assert actual == expected
    assert [_nan_safe(x._properties) for x in actual] == [_nan_safe(x._properties) for x in expected]


@pytest.fixture
def pc():
    return PointCollection.from_points(POINTS)


@pytest.fixture
def fc():
    return FeatureCollection(POINTS)


def test_arrays_equal():
    a = np.array([1.0, np.nan])
    assert _arrays_equal(None, None)
    assert not _arrays_equal(a, None)
    assert not _arrays_equal(a, np.array([1.0]))
    assert _arrays_equal(a, a.copy())

    t = np.array(['2020-01-01', 'NaT'], dtype='datetime64[us]')
    assert _arrays_equal(t, t.copy())
    assert not _arrays_equal(t, np.array(['2020-01-02', 'NaT'], dtype='datetime64[us]'))

    assert _arrays_equal(np.array(['a', None], dtype=object), np.array(['a', None], dtype=object))
    assert not _arrays_equal(np.array(['a', None], dtype=object), np.array(['b', None], dtype=object))


def test_as_column():
    assert _as_column([True, False], 'x', 2).dtype == bool
    assert _as_column([1, 2], 'x', 2).dtype == np.int64
    assert _as_column([1.0, 2.0], 'x', 2).dtype == np.float64

    # Mixed types are not coerced to a common type
    assert _as_column([1, 'a'], 'x', 2).tolist() == [1, 'a']
    assert _as_column([1, 2.0], 'x', 2).tolist() == [1, 2.0]
    assert _as_column([[1, 2], [3, 4]], 'x', 2).tolist() == [[1, 2], [3, 4]]

    # Too large for int64
    assert _as_column([2 ** 70, 1], 'x', 2).tolist() == [2 ** 70, 1]

    # Arrays keep their dtype and are copied
    source = np.array(['a', 'b'])
    col = _as_column(source, 'x', 2)
    assert col.dtype == source.dtype and col is not source

    with pytest.raises(ValueError, match='one-dimensional'):
        _as_column(np.zeros((2, 2)), 'x', 2)

    with pytest.raises(ValueError, match='expected 3'):
        _as_column([1, 2], 'x', 3)


def test_as_float_array():
    with pytest.raises(ValueError, match='one-dimensional'):
        _as_float_array([[1.0]], 'longitude')

    with pytest.raises(ValueError, match='expected 2'):
        _as_float_array([1.0], 'latitude', 2)


def test_as_time_array():
    arr = _as_time_array(np.array(['2020-01-01T00:00'], dtype='datetime64[ns]'), 'dt', 1)
    assert arr.dtype == np.dtype('datetime64[us]')

    mixed = _as_time_array([T0, None, np.datetime64('2020-01-02')], 'dt', 3)
    assert mixed.tolist() == [datetime(2020, 1, 1), None, datetime(2020, 1, 2)]

    # Naive datetimes are taken to be UTC already, as TimeInterval does
    assert _as_time_array([datetime(2020, 1, 1)], 'dt', 1).tolist() == [datetime(2020, 1, 1)]

    with pytest.raises(ValueError, match='one-dimensional'):
        _as_time_array([T0], 'dt', 2)


def test_pointcollection_init():
    pc = PointCollection(
        [1, '2'], [3, 4],
        z=[1.0, np.nan],
        dt_start=[T0, None],
        properties={'a': ['x', 'y']}
    )
    assert pc.longitude.tolist() == [1.0, 2.0]
    assert pc.z.tolist()[0] == 1.0 and np.isnan(pc.z[1])
    assert pc.m is None
    # dt_end defaults to dt_start
    assert _arrays_equal(pc.dt_start, pc.dt_end)

    # Only an end time
    pc = PointCollection([1], [2], dt_end=[T0])
    assert pc[0].dt == TimeInterval(T0, T0)

    # No times
    pc = PointCollection([1], [2])
    assert pc[0].dt is None

    # All-NaN Z/M are dropped
    assert PointCollection([1], [2], z=[np.nan], m=[np.nan]).z is None

    with pytest.raises(ValueError, match='finite'):
        PointCollection([np.nan], [0])

    with pytest.raises(ValueError, match='both a start and an end'):
        PointCollection([1, 2], [1, 2], dt_start=[T0, T0], dt_end=[T0, None])

    with pytest.raises(ValueError, match='earlier than'):
        PointCollection([1], [1], dt_start=[T0], dt_end=[T0 - timedelta(1)])


def test_pointcollection_init_wraps_coordinates():
    lons = [190.0, -190.0, 180.0, 540.0, -900.0, 10.0, 10.0, -10.0, 0.0]
    lats = [0.0, 0.0, 0.0, 0.0, 0.0, 95.0, -95.0, 270.0, 1000.0]
    pc = PointCollection(lons, lats)
    expected = [Coordinate(x, y) for x, y in zip(lons, lats)]
    assert [x.coordinate for x in pc] == expected


def test_pointcollection_bool():
    assert PointCollection.from_points(POINTS)
    assert not PointCollection([], [])


def test_pointcollection_contains(pc):
    assert POINTS[7] in pc
    assert GeoPoint(POINTS[7].coordinate, dt=T0 - timedelta(days=1)) not in pc
    assert POINTS[7].coordinate not in pc

    # Untimed points
    assert POINTS[0] in pc

    zm = PointCollection([1], [1], z=[2], m=[3])
    assert GeoPoint(Coordinate(1, 1, 2, 3)) in zm
    assert GeoPoint(Coordinate(1, 1, 2)) not in zm
    assert GeoPoint(Coordinate(1, 1, m=3)) not in zm
    assert GeoPoint(Coordinate(1, 1, 2, 3)) not in PointCollection([1], [1])


def test_pointcollection_eq(pc):
    assert pc == PointCollection.from_points(POINTS)
    assert pc != PointCollection.from_points(POINTS[:-1])
    assert pc != FeatureCollection(POINTS)

    a = PointCollection([1], [1], properties={'a': [1]})
    assert a != PointCollection([1], [1], properties={'b': [1]})
    assert a != PointCollection([1], [1], properties={'a': [2]})

    partial = PointCollection.from_points([GeoPoint(Coordinate(1, 1), properties={'a': 1}), Coordinate(2, 2)])
    assert partial == PointCollection.from_points(partial.to_points())
    assert partial != PointCollection([1, 2], [1, 2], properties={'a': [1, None]})


def test_pointcollection_getitem(pc):
    assert pc[3] == POINTS[3]
    assert pc[3]._properties == POINTS[3]._properties
    assert pc[np.int64(3)] == POINTS[3]
    assert pc[-1] == POINTS[-1]

    with pytest.raises(IndexError):
        _ = pc[len(POINTS)]

    _assert_same_points(pc[2:10:2], POINTS[2:10:2])
    _assert_same_points(pc[[5, 1]], [POINTS[5], POINTS[1]])

    mask = pc.longitude > 0
    _assert_same_points(pc[mask], [x for x, keep in zip(POINTS, mask) if keep])

    with pytest.raises(TypeError):
        _ = pc['a']


def test_pointcollection_iter(pc):
    _assert_same_points(pc, POINTS)

    zm = PointCollection([1, 2], [1, 2], z=[5, np.nan], m=[np.nan, 6])
    assert [x.coordinate for x in zm] == [Coordinate(1, 1, z=5), Coordinate(2, 2, m=6)]


def test_pointcollection_len(pc):
    assert len(pc) == len(POINTS)


def test_pointcollection_repr(pc):
    assert repr(pc) == f'<PointCollection with {len(POINTS)} points>'
    assert repr(PointCollection([], [])) == '<Empty PointCollection>'


def test_pointcollection_bounds(pc, fc):
    assert pc.bounds == fc.bounds

    with pytest.raises(ValueError):
        _ = PointCollection([], []).bounds


def test_pointcollection_centroid(pc, fc):
    assert pc.centroid == fc.centroid

    with pytest.raises(ValueError):
        _ = PointCollection([], []).centroid


def test_pointcollection_dt_start_end(pc):
    assert pc.dt_start[2] == np.datetime64(POINTS[2].dt.start.replace(tzinfo=None), 'us')
    assert pc.dt_end[2] == np.datetime64(POINTS[2].dt.end.replace(tzinfo=None), 'us')
    assert np.isnat(pc.dt_start[0])


def test_pointcollection_geospan(pc, fc):
    assert pc.geospan == fc.geospan


def test_pointcollection_arrays_are_read_only(pc):
    for arr in (pc.longitude, pc.latitude, pc.dt_start, pc.dt_end, pc.properties['idx']):
        with pytest.raises(ValueError):
            arr[0] = arr[1]

    with pytest.raises(TypeError):
        pc.properties['new'] = np.zeros(len(pc))  # type: ignore[index]

    with pytest.raises(AttributeError):
        pc.anything = 1


def test_pointcollection_properties(pc):
    assert pc.properties['idx'].tolist() == list(range(len(POINTS)))
    assert pc.properties['sometimes'][1] is None


def test_pointcollection_filter_by_dt(pc, fc):
    for dt in (WINDOW, T0 + timedelta(hours=3), T0 + timedelta(hours=3, minutes=30), T0.replace(tzinfo=None)):
        _assert_same_points(pc.filter_by_dt(dt), fc.filter_by_dt(dt))

    with pytest.raises(ValueError):
        pc.filter_by_dt('2020-01-01')  # type: ignore[arg-type]


@pytest.mark.parametrize('name', SHAPES)
@pytest.mark.parametrize('dt', [None, WINDOW, T0 + timedelta(hours=4)])
def test_pointcollection_filter_by_intersection(pc, fc, name, dt):
    shape = _with_dt(SHAPES[name], dt) if dt else SHAPES[name]
    _assert_same_points(pc.filter_by_intersection(shape), fc.filter_by_intersection(shape))


def test_pointcollection_filter_by_property(pc, fc):
    func = lambda x: x == 'even'  # noqa: E731
    _assert_same_points(pc.filter_by_property('parity', func), fc.filter_by_property('parity', func))

    timed = PointCollection.from_points([x for x in POINTS if x.dt])
    after = lambda x: x > T0 + timedelta(hours=4)  # noqa: E731
    _assert_same_points(
        timed.filter_by_property('datetime_end', after),
        FeatureCollection([x for x in POINTS if x.dt]).filter_by_property('datetime_end', after)
    )

    with pytest.raises(KeyError):
        pc.filter_by_property('nonexistent', func)

    with pytest.raises(KeyError):
        # Only some points carry it
        pc.filter_by_property('sometimes', func)

    with pytest.raises(KeyError):
        # Some points have no time
        pc.filter_by_property('datetime_start', func)


@pytest.mark.parametrize('name', SHAPES)
@pytest.mark.parametrize('dt', [None, WINDOW, T0 + timedelta(hours=4)])
def test_pointcollection_filter_contained_by(pc, fc, name, dt):
    shape = _with_dt(SHAPES[name], dt) if dt else SHAPES[name]
    _assert_same_points(pc.filter_contained_by(shape), fc.filter_contained_by(shape))


def test_pointcollection_filter_contained_by_non_haversine(pc, fc):
    try:
        set_geodesic_algorithm('vincenty')
        for name in ('circle', 'circle_hole'):
            _assert_same_points(pc.filter_contained_by(SHAPES[name]), fc.filter_contained_by(SHAPES[name]))
    finally:
        set_geodesic_algorithm('haversine')


class _TimeOnlyEllipse(GeoEllipse):
    """Overrides contains_shape, so PointCollection can't assume it only tests coordinates"""

    def contains_shape(self, shape, **kwargs):
        return super().contains_shape(shape, **kwargs) and shape.dt is not None


@pytest.mark.parametrize('dt', [None, WINDOW])
def test_pointcollection_custom_contains_shape(pc, fc, dt):
    shape = _TimeOnlyEllipse(Coordinate(0, 0), 200_000, 100_000, 30, dt=dt)
    _assert_same_points(pc.filter_contained_by(shape), fc.filter_contained_by(shape))
    _assert_same_points(pc.filter_by_intersection(shape), fc.filter_by_intersection(shape))
    # The override is honored: untimed points are excluded
    assert all(x.dt is not None for x in pc.filter_contained_by(shape))


def test_pointcollection_from_geopandas(fc):
    # FeatureCollection.to_geopandas goes through WKT, which rounds coordinates,
    # so compare against reading the same dataframe back the object way
    df = fc.to_geopandas()
    _assert_same_points(PointCollection.from_geopandas(df), FeatureCollection.from_geopandas(df))

    # With Z values, and only one of the two time fields
    zfc = FeatureCollection([
        GeoPoint(Coordinate(1, 2, z=3), dt=T0),
        GeoPoint(Coordinate(4, 5), dt=T0 + timedelta(hours=1)),
    ])
    df = zfc.to_geopandas().drop(columns=['datetime_end'])
    pc = PointCollection.from_geopandas(df)
    assert pc.z.tolist()[0] == 3.0 and np.isnan(pc.z[1])
    _assert_same_points(pc, zfc)

    # Exact coordinates and times survive a PointCollection round trip (a
    # dataframe has no way to leave a property off some rows, so those come
    # back as NaN, just as with FeatureCollection)
    pc = PointCollection.from_points(POINTS)
    assert PointCollection.from_geopandas(pc.to_geopandas()).to_points() == POINTS

    # A point with only an end time is an instant at that time
    df = zfc.to_geopandas()
    df.loc[0, 'datetime_start'] = None
    assert PointCollection.from_geopandas(df)[0].dt == TimeInterval(T0, T0)

    # No time fields at all
    df = FeatureCollection([GeoPoint(Coordinate(1, 2), properties={'a': 1})]).to_geopandas()
    assert PointCollection.from_geopandas(df)[0]._properties == {'a': 1}

    with pytest.raises(ValueError, match='Point'):
        PointCollection.from_geopandas(FeatureCollection([_HOLE]).to_geopandas())


def test_pointcollection_from_points():
    pc = PointCollection.from_points([Coordinate(1, 2), GeoPoint(Coordinate(3, 4), dt=T0)])
    assert pc[0] == GeoPoint(Coordinate(1, 2))
    assert pc[1] == GeoPoint(Coordinate(3, 4), dt=T0)

    assert len(PointCollection.from_points([])) == 0

    with pytest.raises(TypeError, match='GeoPolygon'):
        PointCollection.from_points([_HOLE])  # type: ignore[list-item]


@pytest.mark.parametrize('name', SHAPES)
def test_pointcollection_intersects(pc, fc, name):
    for dt in (None, WINDOW):
        shape = _with_dt(SHAPES[name], dt) if dt else SHAPES[name]
        assert pc.intersects(shape) == fc.intersects(shape)

    far = GeoBox(Coordinate(50, 51), Coordinate(51, 50))
    assert not pc.intersects(far)


def test_pointcollection_masks(pc):
    shape = SHAPES['polygon']
    mask = pc.mask_contained_by(shape)
    assert mask.dtype == bool and mask.shape == (len(pc),)
    assert mask.sum() == len(pc.filter_contained_by(shape))
    assert pc.mask_by_intersection(shape).sum() == len(pc.filter_by_intersection(shape))
    assert pc.mask_by_dt(WINDOW).sum() == len(pc.filter_by_dt(WINDOW))


def test_pointcollection_to_feature_collection(pc, fc):
    assert pc.to_feature_collection() == fc


def test_pointcollection_to_geopandas(pc, fc):
    df = pc.to_geopandas()
    expected = fc.to_geopandas()
    assert len(df) == len(expected)
    assert set(df.columns) == set(expected.columns)
    for column in ('idx', 'parity', 'sometimes'):
        # Same values and the same inferred dtype
        assert df[column].equals(expected[column])
    assert df['datetime_start'].equals(expected['datetime_start'].astype(df['datetime_start'].dtype))
    # FeatureCollection goes through WKT, which rounds; PointCollection is exact
    assert shapely.equals_exact(df.geometry.values, expected.geometry.values, tolerance=1e-9).all()

    assert list(pc.to_geopandas(include_properties=['idx', 'missing']).columns) == ['idx', 'missing', 'geometry']

    # Untimed points don't get time columns
    untimed = PointCollection([1], [2], z=[3])
    df = untimed.to_geopandas()
    assert list(df.columns) == ['geometry']
    assert df.geometry[0].has_z


def test_pointcollection_to_points(pc):
    _assert_same_points(pc.to_points(), POINTS)


def test_pointcollection_to_shapely():
    assert shapely.equals(PointCollection([1], [2]).to_shapely(), shapely.points([1], [2])).all()

    with_z = PointCollection([1], [2], z=[3]).to_shapely()
    assert shapely.has_z(with_z).all()


def test_pointcollection_to_track():
    timed = [x for x in POINTS if x.dt]
    assert PointCollection.from_points(timed).to_track() == Track(timed)

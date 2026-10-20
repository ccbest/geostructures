"""
The columnar fast paths must return exactly what the per-shape methods would.
These tests compare the two directly over randomized collections built to
exercise boundaries: shared time bounds, points on polygon vertices and
edges, the antimeridian, and the poles.
"""
from datetime import datetime, timedelta, timezone
import math
import random

import numpy as np
import pytest

from geostructures import (
    Coordinate, FeatureCollection, GeoBox, GeoCircle, GeoEllipse, GeoLineString,
    GeoPoint, GeoPolygon, GeoRing, MultiGeoPoint, MultiGeoPolygon, Track
)
from geostructures._columnar import ShapeColumns, points_in_shape, shape_envelope
from geostructures.geodesic import set_geodesic_algorithm
from geostructures.time import TimeInterval


# A handful of shared instants so that interval bounds frequently coincide
_TIMES = [datetime(2020, 1, 1, tzinfo=timezone.utc) + timedelta(hours=h) for h in range(6)]


def _random_dt(rnd):
    roll = rnd.random()
    if roll < 0.3:
        return None
    if roll < 0.55:
        return rnd.choice(_TIMES)
    start, end = sorted(rnd.sample(_TIMES, 2))
    return TimeInterval(start, end)


def _ring(rnd, lon, lat, size, n):
    angles = sorted(rnd.uniform(0, 2 * math.pi) for _ in range(n))
    coords = [
        Coordinate(lon + size * rnd.uniform(0.3, 1) * math.cos(a), lat + size * rnd.uniform(0.3, 1) * math.sin(a))
        for a in angles
    ]
    return [*coords, coords[0]]


def _random_polygon(rnd, lon, lat, size, dt=None, holes=False):
    hole_shapes = None
    if holes:
        hole_shapes = [GeoPolygon(_ring(rnd, lon, lat, size * 0.2, 4), _is_hole=True)]
    return GeoPolygon(_ring(rnd, lon, lat, size, rnd.randint(3, 8)), holes=hole_shapes, dt=dt)


def _random_center(rnd):
    roll = rnd.random()
    if roll < 0.15:
        # Near the antimeridian
        return rnd.choice([-1, 1]) * rnd.uniform(178, 179.9), rnd.uniform(-10, 10)
    if roll < 0.25:
        # Near a pole
        return rnd.uniform(-180, 179), rnd.choice([-1, 1]) * rnd.uniform(85, 89.5)
    return rnd.uniform(-5, 5), rnd.uniform(-5, 5)


def _random_shapes(rnd, n):
    shapes = []
    anchors = []  # polygons whose vertices/edges also seed points
    for _ in range(n):
        lon, lat = _random_center(rnd)
        dt = _random_dt(rnd)
        kind = rnd.random()
        if kind < 0.45:
            if anchors and rnd.random() < 0.5:
                # On a vertex or edge of an earlier polygon
                ring = rnd.choice(anchors).outline
                i = rnd.randrange(len(ring) - 1)
                a, b = ring[i], ring[i + 1]
                t = rnd.choice([0, 0.5])
                coord = Coordinate(a.longitude + t * (b.longitude - a.longitude), a.latitude + t * (b.latitude - a.latitude))
            else:
                coord = Coordinate(lon + rnd.uniform(-2, 2), lat + rnd.uniform(-2, 2))
            shapes.append(GeoPoint(coord, dt=dt))
        elif kind < 0.6:
            poly = _random_polygon(rnd, lon, lat, rnd.uniform(0.5, 3), dt=dt, holes=rnd.random() < 0.3)
            anchors.append(poly)
            shapes.append(poly)
        elif kind < 0.67:
            # Boxes may not span the antimeridian
            lon = max(-178., min(178., lon))
            shapes.append(GeoBox(Coordinate(lon - 1, lat + 0.5), Coordinate(lon + 1, lat - 0.5), dt=dt))
        elif kind < 0.75:
            shapes.append(GeoCircle(Coordinate(lon, lat), rnd.uniform(1_000, 300_000), dt=dt))
        elif kind < 0.8:
            shapes.append(GeoEllipse(Coordinate(lon, lat), 200_000, 50_000, rnd.uniform(0, 180), dt=dt))
        elif kind < 0.85:
            shapes.append(GeoRing(Coordinate(lon, lat), 50_000, 150_000, dt=dt))
        elif kind < 0.93:
            shapes.append(GeoLineString(
                [Coordinate(lon + rnd.uniform(-2, 2), lat + rnd.uniform(-2, 2)) for _ in range(3)],
                dt=dt
            ))
        elif kind < 0.97:
            shapes.append(MultiGeoPoint(
                [GeoPoint(Coordinate(lon + rnd.uniform(-1, 1), lat + rnd.uniform(-1, 1))) for _ in range(2)],
                dt=dt
            ))
        else:
            shapes.append(MultiGeoPolygon(
                [_random_polygon(rnd, lon, lat, 0.5), _random_polygon(rnd, lon + 1.5, lat, 0.5)],
                dt=dt
            ))
    return shapes, anchors


def _queries(rnd, shapes, anchors):
    queries = []
    for dt in (None, TimeInterval(_TIMES[1], _TIMES[3]), _TIMES[2]):
        queries += [
            _random_polygon(rnd, 0, 0, 4, dt=dt),
            _random_polygon(rnd, 0, 0, 4, dt=dt, holes=True),
            # Crosses the antimeridian
            GeoPolygon([
                Coordinate(178., -5.), Coordinate(-178., -5.), Coordinate(-178., 5.),
                Coordinate(178., 5.), Coordinate(178., -5.)
            ], dt=dt),
            GeoBox(Coordinate(-3., 3.), Coordinate(3., -3.), dt=dt),
            # A circular hole has no vectorized point test, forcing the fallback
            GeoBox(Coordinate(-3., 3.), Coordinate(3., -3.), holes=[GeoCircle(Coordinate(0., 0.), 50_000)], dt=dt),
            GeoCircle(Coordinate(0., 0.), 300_000, dt=dt),
            GeoCircle(Coordinate(179.5, 0.), 200_000, dt=dt),
            GeoCircle(Coordinate(0., 88.), 400_000, dt=dt),
            GeoLineString([Coordinate(-4., -4.), Coordinate(4., 4.)], dt=dt),
            MultiGeoPolygon([_random_polygon(rnd, -2, 0, 1), _random_polygon(rnd, 2, 0, 1)], dt=dt),
        ]
        # Shapes from the collection itself, including exact duplicates of points
        queries += rnd.sample(shapes, 5)
        queries += [x.copy() for x in rnd.sample(anchors, min(2, len(anchors)))]
    return queries


@pytest.fixture(params=['haversine', 'vincenty', 'karney'])
def geodesic_algorithm(request):
    set_geodesic_algorithm(request.param)
    yield request.param
    set_geodesic_algorithm('haversine')


def _ids(collection):
    return [id(x) for x in collection]


# The ellipsoidal algorithms only change curved-shape extents, which
# test_curved_shape_envelopes_contain_shape covers for all of them; they are
# slow, so one ellipsoidal run suffices here
@pytest.mark.parametrize(
    'geodesic_algorithm, seed',
    [('haversine', 0), ('haversine', 1), ('haversine', 2), ('vincenty', 0)],
    indirect=['geodesic_algorithm'],
)
def test_spatial_filters_match_per_shape(geodesic_algorithm, seed):
    rnd = random.Random(seed)
    shapes, anchors = _random_shapes(rnd, 120)
    fcol = FeatureCollection(shapes)

    for query in _queries(rnd, shapes, anchors):
        assert _ids(fcol.filter_by_intersection(query)) == \
            [id(x) for x in shapes if x.intersects(query)], query
        assert _ids(fcol.filter_contained_by(query)) == \
            [id(x) for x in shapes if query.contains(x)], query
        assert _ids(fcol.filter_contains(query)) == \
            [id(x) for x in shapes if x.contains(query)], query
        assert fcol.intersects(query) == any(x.intersects(query) for x in shapes), query


@pytest.mark.parametrize('seed', range(5))
def test_time_filters_match_per_shape(seed):
    rnd = random.Random(seed)
    shapes, _ = _random_shapes(rnd, 300)
    fcol = FeatureCollection(shapes)

    queries = [TimeInterval(a, b) for a in _TIMES for b in _TIMES if a <= b]
    for query in queries:
        assert _ids(fcol.filter_by_dt(query)) == \
            [id(x) for x in shapes if x.dt is not None and query.intersects(x.dt)]

    timed = [x for x in shapes if x.dt is not None]
    track = Track(timed)
    for instant in _TIMES:
        assert _ids(track[instant]) == \
            [id(x) for x in track if TimeInterval(instant, instant).intersects(x.dt)]
        assert _ids(track[instant:]) == [id(x) for x in track if instant <= x.start]
        assert _ids(track[:instant]) == [id(x) for x in track if x.end < instant]
        for stop in _TIMES:
            assert _ids(track[instant:stop]) == \
                [id(x) for x in track if instant <= x.start and x.end < stop]


@pytest.mark.parametrize('seed', range(5))
def test_points_in_shape_matches_contains_coordinate(seed):
    rnd = random.Random(seed)
    polygons = [
        _random_polygon(rnd, 0, 0, 3, holes=rnd.random() < 0.5) for _ in range(10)
    ] + [
        GeoPolygon([
            Coordinate(178., -5.), Coordinate(-178., -5.), Coordinate(-178., 5.),
            Coordinate(178., 5.), Coordinate(178., -5.)
        ]),
        GeoBox(Coordinate(-2., 2.), Coordinate(2., -2.), holes=[GeoBox(Coordinate(-1., 1.), Coordinate(1., -1.))]),
    ]
    for shape in polygons:
        # Random points, plus every vertex, edge midpoint and vertex latitude
        coords = [Coordinate(rnd.uniform(-180, 180), rnd.uniform(-6, 6)) for _ in range(300)]
        coords += [Coordinate(rnd.uniform(-4, 4), rnd.uniform(-4, 4)) for _ in range(300)]
        for ring in shape.linear_rings():
            for a, b in zip(ring, ring[1:]):
                coords += [
                    a,
                    Coordinate((a.longitude + b.longitude) / 2, (a.latitude + b.latitude) / 2),
                    Coordinate(a.longitude + 0.1, a.latitude),
                    Coordinate(a.longitude - 0.1, a.latitude),
                ]

        lons = np.array([c.longitude for c in coords])
        lats = np.array([c.latitude for c in coords])
        assert points_in_shape(shape, lons, lats).tolist() == \
            [shape.contains_coordinate(c) for c in coords]


def test_points_in_shape_unsupported():
    lons, lats = np.zeros(1), np.zeros(1)
    assert points_in_shape(GeoCircle(Coordinate(0., 0.), 1_000), lons, lats) is None
    box = GeoBox(Coordinate(-1., 1.), Coordinate(1., -1.), holes=[GeoCircle(Coordinate(0., 0.), 1_000)])
    assert points_in_shape(box, lons, lats) is None


@pytest.mark.parametrize('center', [(0., 0.), (0., 60.), (179.5, 0.), (10., 89.)])
def test_curved_shape_envelopes_contain_shape(center, geodesic_algorithm):
    center = Coordinate(*center)
    shapes = [
        GeoCircle(center, 150_000),
        GeoEllipse(center, 150_000, 40_000, 30),
        GeoRing(center, 50_000, 150_000),
        GeoRing(center, 50_000, 150_000, angle_min=10, angle_max=100),
    ]
    rnd = random.Random(0)
    for shape in shapes:
        min_lon, min_lat, max_lon, max_lat = shape_envelope(shape)
        boundary = shape.bounding_coords(k=360)
        nearby = [
            Coordinate(center.longitude + rnd.uniform(-3, 3), center.latitude + rnd.uniform(-2, 2))
            for _ in range(400)
        ]
        for coord in [*boundary, *(x for x in nearby if x in shape)]:
            assert min_lon <= coord.longitude <= max_lon, (shape, coord)
            assert min_lat <= coord.latitude <= max_lat, (shape, coord)


def test_shape_envelope_unknown_shape_spans_globe():
    class CustomPoint(GeoPoint):
        pass

    assert shape_envelope(CustomPoint(Coordinate(0., 0.))) == \
        (-math.inf, -math.inf, math.inf, math.inf)


def test_shape_columns_take_carries_computed_arrays():
    shapes = [
        GeoPoint(Coordinate(0., 0.), dt=datetime(2020, 1, 1)),
        GeoBox(Coordinate(0., 1.), Coordinate(1., 0.)),
        GeoPoint(Coordinate(1., 1.)),
    ]
    columns = ShapeColumns(shapes)
    _ = columns.has_dt

    subset = columns.take(np.array([0, 2]), [shapes[0], shapes[2]])
    assert len(subset) == 2
    assert '_times' in subset.__dict__
    assert subset.has_dt.tolist() == [True, False]

    # Not yet computed on the parent, so computed on demand from the subset
    assert 'envelopes' not in subset.__dict__ and '_points' not in subset.__dict__
    assert subset.is_point.tolist() == [True, True]
    np.testing.assert_array_equal(subset.envelopes, columns.envelopes[[0, 2]])


def test_filtered_collections_reuse_columns():
    fcol = FeatureCollection([
        GeoPoint(Coordinate(float(i), 0.), dt=datetime(2020, 1, 1)) for i in range(5)
    ])
    box = GeoBox(Coordinate(0.5, 1.), Coordinate(3.5, -1.))

    filtered = fcol.filter_by_intersection(box)
    assert [x.centroid.longitude for x in filtered] == [1., 2., 3.]
    assert '_columns' in filtered.__dict__
    assert filtered.filter_by_dt(datetime(2020, 1, 1)) == filtered


def test_envelope_cached_on_shape():
    import pickle

    box = GeoBox(Coordinate(0.0, 1.0), Coordinate(1.0, 0.0))
    envelope = shape_envelope(box)
    assert box.__dict__['_envelope'] == envelope
    assert '_envelope' not in pickle.loads(pickle.dumps(box)).__dict__

    # Discarded with the shape's other cached values (deprecated mutation)
    box.set_dt(datetime(2020, 1, 1), inplace=True)
    assert '_envelope' not in box.__dict__
    box.nw_bound = Coordinate(5.0, 6.0)
    box.se_bound = Coordinate(6.0, 5.0)
    assert shape_envelope(box)[0] > 4.0

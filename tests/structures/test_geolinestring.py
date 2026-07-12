
from datetime import datetime

import pytest
from shapely import wkt

from geostructures import GeoBox, GeoCircle, GeoLineString, GeoPoint, GeoPolygon, MultiGeoPoint
from geostructures.coordinates import Coordinate
from geostructures.geodesic import destination_point, distance_meters

from tests.functions import (
    assert_geolinestrings_equal, default_test_datetime,
    geojson_round_trip, shapely_round_trip, wkt_round_trip,
)


def test_geolinestring_eq():
    ls = GeoLineString(
        [Coordinate(0.0, 0.0), Coordinate(1.0, 0.0), Coordinate(1.0, 1.0)],
        dt=default_test_datetime
    )
    l2 = GeoLineString(
        [Coordinate(0.0, 0.0), Coordinate(1.0, 0.0), Coordinate(1.0, 1.0)],
        dt=default_test_datetime
    )
    assert ls == l2

    l2 = GeoLineString(
        [Coordinate(0.0, 0.0), Coordinate(1.0, 0.0), Coordinate(2.0, 1.0)],
        dt=default_test_datetime
    )
    assert ls != l2

    l2 = GeoLineString(
        [Coordinate(0.0, 0.0), Coordinate(1.0, 0.0), Coordinate(1.0, 1.0)],
        dt=datetime(1970, 1, 1, 1, 1)
    )
    assert ls != l2

    assert 'test' != ls


def test_geolinestring_hash():
    ls = GeoLineString(
        [Coordinate(0.0, 0.0), Coordinate(1.0, 0.0), Coordinate(1.0, 1.0)],
        dt=default_test_datetime
    )
    l2 = GeoLineString(
        [Coordinate(0.0, 0.0), Coordinate(1.0, 0.0), Coordinate(1.0, 1.0)],
        dt=default_test_datetime
    )
    assert len({ls, l2}) == 1


def test_geolinestring_repr():
    ls = GeoLineString([Coordinate(0.0, 0.0), Coordinate(1.0, 0.0), Coordinate(1.0, 1.0)])
    assert repr(ls) == '<GeoLineString with 3 points>'


def test_geolinestring_bounds():
    ls = GeoLineString([Coordinate(0.0, 0.0), Coordinate(1.0, 0.0), Coordinate(1.0, 1.0)])
    assert ls.bounds == (
        0., 0., 1., 1.
    )

    # Regression: Z/M-carrying vertices previously crashed bounds
    ls = GeoLineString([
        Coordinate(0.0, 0.0, z=1., m=2.), Coordinate(1.0, 0.0, z=1.), Coordinate(1.0, 1.0, z=1.),
    ])
    assert ls.bounds == (0., 0., 1., 1.)
    assert ls.circumscribing_rectangle() == GeoBox(Coordinate(0., 1.), Coordinate(1., 0.))


def test_geolinestring_length():
    c1, c2, c3 = Coordinate(0.0, 0.0), Coordinate(1.0, 0.0), Coordinate(1.0, 1.0)
    ls = GeoLineString([c1, c2, c3])
    expected = distance_meters(c1, c2) + distance_meters(c2, c3)
    assert ls.length == expected


def test_geolinestring_centroid():
    ls = GeoLineString([Coordinate(0.0, 0.0), Coordinate(1.0, 0.0), Coordinate(1.0, 1.0)])
    assert ls.centroid == Coordinate(0.6666667, 0.3333333)

    # Regression: Z/M-carrying vertices previously crashed centroid
    ls = GeoLineString([
        Coordinate(0.0, 0.0, z=1., m=2.), Coordinate(1.0, 0.0, z=1.), Coordinate(1.0, 1.0, z=1.),
    ])
    assert ls.centroid == Coordinate(0.6666667, 0.3333333)


def test_geolinestring_contains_dunder():
    ls = GeoLineString([Coordinate(0.0, 0.0), Coordinate(1.0, 0.0), Coordinate(1.0, 1.0)])
    assert Coordinate(0., 0.) in ls
    assert Coordinate(5., 5.) not in ls


def test_geolinestring_contains():
    ls = GeoLineString([Coordinate(0., 0.), Coordinate(1., 1.), Coordinate(2., 2.)], dt=datetime(2020, 1, 1))

    ls2 = GeoCircle(Coordinate(0., 0.), 500)
    assert not ls.contains(ls2)

    ls2 = GeoLineString([Coordinate(0., 0.), Coordinate(1., 1.)])
    assert ls.contains(ls2)

    ls2 = GeoLineString([Coordinate(0., 0.), Coordinate(1., 1.)], dt=datetime(2020, 1, 2))
    assert not ls.contains(ls2)

    ls2 = GeoLineString([Coordinate(0., 1.), Coordinate(0., 0.), Coordinate(1., 1.)])
    assert not ls.contains(ls2)

    point = GeoPoint(Coordinate(0., 0.))
    assert ls.contains(point)

    point2 = GeoPoint(Coordinate(1., 0.))
    assert not ls.contains(point2)

    assert ls.contains_shape(MultiGeoPoint([GeoPoint(Coordinate(0., 0.))]))
    assert not ls.contains_shape(MultiGeoPoint([GeoPoint(Coordinate(5., 0.))]))


def test_geolinestring_intersects_shape():
    ls = GeoLineString([Coordinate(0.0, 0.0), Coordinate(1.0, 1.0)])
    circle = GeoCircle(Coordinate(0.0, 0.0), 5_000)
    assert ls.intersects_shape(circle)

    assert ls.intersects_shape(ls)

    # Contains point
    assert ls.intersects_shape(GeoPoint(Coordinate(0., 0.)))

    assert ls.intersects_shape(MultiGeoPoint([GeoPoint(Coordinate(0., 0.))]))
    assert not ls.intersects_shape(MultiGeoPoint([GeoPoint(Coordinate(5., 0.))]))


def test_geolinestring_copy():
    linestring = GeoLineString([Coordinate(0., 1.), Coordinate(0., 1.)])
    linestring_copy = linestring.copy()

    # Assert equality but different pointer
    assert linestring == linestring_copy
    assert linestring is not linestring_copy


def test_geolinestring_serialization_round_trips():
    ls = GeoLineString(
        [Coordinate(0.0, 0.0), Coordinate(1.0, 0.0), Coordinate(1.0, 1.0)],
        dt=default_test_datetime
    )
    wkt_round_trip(ls)
    geojson_round_trip(ls)
    shapely_round_trip(ls)


def test_geolinestring_to_wkt():
    ls = GeoLineString([Coordinate(0.0, 0.0), Coordinate(1.0, 0.0), Coordinate(1.0, 1.0)])
    assert ls.to_wkt() == 'LINESTRING(0 0,1 0,1 1)'

    wkt.loads(ls.to_wkt())


def test_geolinestring_from_wkt():
    wkt_str = 'LINESTRING (30.123 10, 10 30.123, 40 40)'
    assert GeoLineString.from_wkt(wkt_str) == GeoLineString([
        Coordinate(30.123, 10), Coordinate(10, 30.123), Coordinate(40, 40)
    ])

    wkt_str = 'LINESTRING(30 10,10 30,40 40)'
    assert GeoLineString.from_wkt(wkt_str) == GeoLineString([
        Coordinate(30, 10), Coordinate(10, 30), Coordinate(40, 40)
    ])

    with pytest.raises(ValueError):
        _ = GeoLineString.from_wkt('BAD WKT')

    with pytest.raises(ValueError):
        _ = GeoLineString.from_wkt('LINESTRING(30 10,10 30,40 40) (1 2, 3 4)')


def test_geolinestring_from_geojson():
    gls = {
        'type': 'Feature',
        'geometry': {
            'type': 'LineString',
            'coordinates': [[0.0, 0.0], [1.0, 1.5], [2.0, 2.0]]
        },
        'properties': {'example': 'prop'}
    }
    expected = GeoLineString(
        [Coordinate(0.0, 0.0), Coordinate(1.0, 1.5), Coordinate(2.0, 2.0)],
        properties={'example': 'prop'}
    )
    assert GeoLineString.from_geojson(gls) == expected

    gls = gls['geometry']
    expected = GeoLineString(
        [Coordinate(0.0, 0.0), Coordinate(1.0, 1.5), Coordinate(2.0, 2.0)],
    )
    assert GeoLineString.from_geojson(gls) == expected

    with pytest.raises(ValueError):
        bad_gjson = {
            'type': 'Feature',
            'geometry': {
                'type': 'Error',
                'coordinates': [
                    [0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [0.0, 0.0],
                ]
            },
            'properties': {'example': 'prop'}
        }
        GeoLineString.from_geojson(bad_gjson)


def test_geolinestring_to_geojson():
    ls = GeoLineString(
        [Coordinate(0.0, 0.0), Coordinate(1.0, 0.0), Coordinate(1.0, 1.0)],
        dt=default_test_datetime
    )
    # Assert kwargs and properties end up in the right place
    assert ls.to_geojson(properties={'test_prop': 2}, test_kwarg=1) == {
        'type': 'Feature',
        'geometry': {
            'type': 'LineString',
            'coordinates': [list(x.to_float()) for x in ls.vertices],
        },
        'properties': {
            'test_prop': 2,
            'datetime_start': default_test_datetime.isoformat(),
            'datetime_end': default_test_datetime.isoformat(),
        },
        'test_kwarg': 1
    }


def test_geolinestring_split_by_len():
    c1 = Coordinate(0.0, 0.0)
    c2 = destination_point(c1, 90., 1000)
    c3 = destination_point(c2, 0., 1000)

    c1_mid = destination_point(c1, 90, 750)
    c2_mid = destination_point(c2, 0, 500)

    ls = GeoLineString([c1, c2, c3])
    expected = [
        GeoLineString([c1, c1_mid]),
        GeoLineString([c1_mid, c2, c2_mid]),
        GeoLineString([c2_mid, c3]),
    ]
    actual = ls.split_by_length(750)

    assert len(actual) == len(expected)
    for l1, l2 in zip(actual, expected):
        assert_geolinestrings_equal(l1, l2)

    with pytest.raises(ValueError):
        ls.split_by_length(0)

    with pytest.raises(ValueError):
        ls.split_by_length(-5)


def test_geolinestring_circumscribing_circle():
    ls = GeoLineString(
        [Coordinate(0.0, 0.0), Coordinate(1.0, 0.0), Coordinate(1.0, 1.0)],
        dt=default_test_datetime
    )
    assert ls.circumscribing_circle() == GeoCircle(
        Coordinate(0.6666667, 0.3333333),
        82879.43253850673,
        dt=default_test_datetime
    )


def test_geolinestring_circumscribing_rectangle():
    ls = GeoLineString(
        [Coordinate(0.0, 0.0), Coordinate(1.0, 0.0), Coordinate(1.0, 1.0)],
        dt=default_test_datetime
    )
    assert ls.circumscribing_rectangle() == GeoBox(
        Coordinate(0.0, 1.0),
        Coordinate(1.0, 0.0),
        dt=default_test_datetime
    )


def test_geolinestring_to_polygon():
    ls = GeoLineString(
        [Coordinate(0.0, 0.0), Coordinate(1.0, 0.0), Coordinate(1.0, 1.0)],
        dt=default_test_datetime
    )
    assert ls.to_polygon() == GeoPolygon(
        [Coordinate(0.0, 0.0), Coordinate(1.0, 0.0), Coordinate(1.0, 1.0), Coordinate(0.0, 0.0)],
        dt=default_test_datetime
    )

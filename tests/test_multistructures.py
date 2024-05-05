
from datetime import datetime

import pytest
import shapely

from geostructures import *
from geostructures.multistructures import *
from geostructures.time import TimeInterval


def test_multigeolinestring_contains():
    mls = MultiGeoLineString(
        [
            GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5), Coordinate(1., 0.)]),
            GeoLineString([Coordinate(1., 1.), Coordinate(0.5, 0.5), Coordinate(0., 0.)]),
        ],
        dt=TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 2))
    )
    assert Coordinate(0.5, 0.5) in mls
    assert GeoPoint(Coordinate(0.5, 0.5)) in mls
    assert GeoPoint(Coordinate(0.5, 0.5), dt=datetime(2020, 1, 1, 1))


def test_multigeolinestring_hash():
    mls_1 = MultiGeoLineString(
        [
            GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5), Coordinate(1., 0.)]),
            GeoLineString([Coordinate(1., 1.), Coordinate(0.5, 0.5), Coordinate(0., 0.)]),
        ],
        dt=TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 2))
    )
    mls_2 = MultiGeoLineString(
        [
            GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5), Coordinate(1., 0.)]),
            GeoLineString([Coordinate(1., 1.), Coordinate(0.5, 0.5), Coordinate(0., 0.)]),
        ],
        dt=TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 2))
    )
    assert hash(mls_1) == hash(mls_2)

    mls_2 = MultiGeoLineString(
        [
            GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5), Coordinate(1., 0.)]),
            GeoLineString([Coordinate(1., 1.), Coordinate(0.5, 0.5), Coordinate(0., 0.)]),
        ],
        dt=TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 3))
    )
    assert hash(mls_1) != hash(mls_2)


def test_multigeolinestring_repr():
    mls = MultiGeoLineString(
        [
            GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5), Coordinate(1., 0.)]),
        ],
    )
    assert repr(mls) == '<MultiGeoLineString of 1 linestring>'

    mls = MultiGeoLineString(
        [
            GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5), Coordinate(1., 0.)]),
            GeoLineString([Coordinate(1., 1.), Coordinate(0.5, 0.5), Coordinate(0., 0.)]),
        ],
    )
    assert repr(mls) == '<MultiGeoLineString of 2 linestrings>'


def test_multigeolinestring_bounds():
    mls = MultiGeoLineString(
        [
            GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5), Coordinate(1., 0.)]),
            GeoLineString([Coordinate(2., 1.), Coordinate(0.5, 0.5), Coordinate(0., 0.)]),
        ],
    )
    assert mls.bounds() == ((0., 2.), (0., 1.))


def test_multigeolinestring_circumscribing_circle():
    mls = MultiGeoLineString(
        [
            GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5), Coordinate(1., 0.)]),
            GeoLineString([Coordinate(1., 1.), Coordinate(0.5, 0.5), Coordinate(0., 0.)]),
        ],
    )
    assert mls.circumscribing_circle() == GeoCircle(Coordinate(0.5, 0.5), 78626.18767687456)


def test_multigeolinestring_contains_coordinate():
    mls = MultiGeoLineString(
        [
            GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5), Coordinate(1., 0.)]),
            GeoLineString([Coordinate(1., 1.), Coordinate(0.5, 0.5), Coordinate(0., 0.)]),
        ],
    )
    assert mls.contains_coordinate(Coordinate(0.5, 0.5))
    assert not mls.contains_coordinate(Coordinate(2.0, 2.0))


def test_multigeolinestring_contains_shape():
    mls = MultiGeoLineString(
        [
            GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5), Coordinate(1., 0.)]),
            GeoLineString([Coordinate(1., 1.), Coordinate(0.5, 0.5), Coordinate(0., 0.)]),
        ],
    )
    assert mls.contains_shape(
        MultiGeoLineString(
            [
                GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5)]),
                GeoLineString([Coordinate(1., 1.), Coordinate(0.5, 0.5)]),
            ],
        )
    )
    assert mls.contains_shape(
        MultiGeoPoint([GeoPoint(Coordinate(0., 1.)), GeoPoint(Coordinate(1., 1.))])
    )

    assert not mls.contains_shape(GeoCircle(Coordinate(0., 0.), 500))
    assert not mls.contains_shape(GeoPoint(Coordinate(2., 2.)))


def test_multigeolinestring_copy():
    mls_1 = MultiGeoLineString(
        [
            GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5), Coordinate(1., 0.)]),
            GeoLineString([Coordinate(1., 1.), Coordinate(0.5, 0.5), Coordinate(0., 0.)]),
        ],
    )
    mls_2 = mls_1.copy()
    assert mls_1 == mls_2

    mls_2.geoshapes.pop(0)
    assert mls_1 != mls_2


def test_multigeolinestring_from_geojson():
    gjson = {
        "type": "Feature",
        "geometry": {
            "type": "MultiLineString",
            "coordinates": [
                [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]],
                [[1.0, 1.0], [0.5, 0.5], [0.0, 0.0]]
            ]
        },
        "properties": {
            "test_prop": "test_value",
            "datetime_start": "2020-01-01 00:00:00.000",
            "datetime_end": "2020-01-02 00:00:00.000",
        }
    }
    mls = MultiGeoLineString.from_geojson(gjson)
    assert mls.geoshapes == [
        GeoLineString([Coordinate(0.0, 1.0), Coordinate(0.5, 0.5), Coordinate(1.0, 0.0)]),
        GeoLineString([Coordinate(1.0, 1.0), Coordinate(0.5, 0.5), Coordinate(0.0, 0.0)])
    ]
    assert mls.dt == TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 2))
    assert mls._properties == {'test_prop': 'test_value'}


def test_multigeolinestring_from_shapely():
    smls = shapely.MultiLineString([[[0, 0], [1, 2]], [[4, 4], [5, 6]]])
    mls = MultiGeoLineString.from_shapely(smls)
    expected = [
        GeoLineString([Coordinate(0.0, 0.0), Coordinate(1., 2.)]),
        GeoLineString([Coordinate(4.0, 4.0), Coordinate(5., 6.)])
    ]
    assert all([a == b for a, b in zip(mls.geoshapes, expected)])


def test_multigeolinestring_from_wkt():
    wkt = "MULTILINESTRING((0.0 1.0,0.5 0.5,1.0 0.0), (1.0 1.0,0.5 0.5,0.0 0.0))"
    mls = MultiGeoLineString.from_wkt(wkt)
    assert mls.geoshapes == [
        GeoLineString([Coordinate(0.0, 1.0), Coordinate(0.5, 0.5), Coordinate(1.0, 0.0)]),
        GeoLineString([Coordinate(1.0, 1.0), Coordinate(0.5, 0.5), Coordinate(0.0, 0.0)])
    ]


def test_multigeolinestring_intersects_shape():
    pass
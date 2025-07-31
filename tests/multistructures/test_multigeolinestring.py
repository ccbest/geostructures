from datetime import datetime

import pytest
import shapely

from geostructures import FeatureCollection, MultiGeoLineString, GeoLineString, Coordinate, GeoCircle, GeoPolygon
from geostructures.time import TimeInterval

from tests.functions import pyshp_round_trip


def test_multigeolinestring_geo_interface():
    mls = MultiGeoLineString([
        GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5), Coordinate(1., 0.)]),
        GeoLineString([Coordinate(1., 1.), Coordinate(0.5, 0.5), Coordinate(0., 0.)]),
    ])
    assert mls.__geo_interface__ == {
        'type': 'MultiLineString',
        'coordinates': [
            [[0., 1.], [0.5, 0.5], [1., 0.]],
            [[1., 1.], [0.5, 0.5], [0., 0.]]
        ]
    }


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


def test_multigeolinestring_centroid():
    mls = MultiGeoLineString(
        [
            GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5), Coordinate(1., 0.)]),
            GeoLineString([Coordinate(1., 1.), Coordinate(0.5, 0.5), Coordinate(0., 0.)]),
        ],
    )
    assert mls.centroid == Coordinate(0.5, 0.5)


def test_multigeolinestring_segments():
    mls = MultiGeoLineString(
        [
            GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5), Coordinate(1., 0.)]),
            GeoLineString([Coordinate(1., 1.), Coordinate(0.5, 0.5), Coordinate(0., 0.)]),
        ],
    )
    assert mls.segments == [
        [(Coordinate(0., 1.), Coordinate(0.5, 0.5)), (Coordinate(0.5, 0.5), Coordinate(1., 0.))],
        [(Coordinate(1., 1.), Coordinate(0.5, 0.5)), (Coordinate(0.5, 0.5), Coordinate(0., 0.))]
    ]


def test_multigeolinestring_circumscribing_circle():
    mls = MultiGeoLineString(
        [
            GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5), Coordinate(1., 0.)]),
            GeoLineString([Coordinate(1., 1.), Coordinate(0.5, 0.5), Coordinate(0., 0.)]),
        ],
    )
    assert mls.circumscribing_circle() == GeoCircle(Coordinate(0.5, 0.5), 78626.18767687456)


def test_multigeolinestring_convex_hull():
    mp = MultiGeoLineString([
        GeoLineString([Coordinate(0., 1.), Coordinate(1., 0.)]),
        GeoLineString([Coordinate(0., 0.), Coordinate(1., 1.)]),
    ])
    assert mp.convex_hull() == GeoPolygon([
        Coordinate(0., 0.),
        Coordinate(1., 0.),
        Coordinate(1., 1.),
        Coordinate(0., 1.),
        Coordinate(0., 0.),
    ])


def test_multigeolinestring_copy():
    mls = MultiGeoLineString(
        [
            GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5), Coordinate(1., 0.)]),
            GeoLineString([Coordinate(1., 1.), Coordinate(0.5, 0.5), Coordinate(0., 0.)]),
        ],
    )
    mls2 = mls.copy()
    assert mls == mls2


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

    # Only geo interface
    gjson = gjson['geometry']
    mls = MultiGeoLineString.from_geojson(gjson)
    assert mls.geoshapes == [
        GeoLineString([Coordinate(0.0, 1.0), Coordinate(0.5, 0.5), Coordinate(1.0, 0.0)]),
        GeoLineString([Coordinate(1.0, 1.0), Coordinate(0.5, 0.5), Coordinate(0.0, 0.0)])
    ]

    with pytest.raises(ValueError):
        gjson = {
            "type": "Feature",
            "geometry": {
                "type": "SomethingElse",
            }
        }
        MultiGeoLineString.from_geojson(gjson)


def test_multigeolinestring_from_pyshp():
    class MockShape:
        def __init__(self):
            self.z = [10., 12., 14., 16.]
            self.m = [11., 13., 15., 17.]

        @property
        def __geo_interface__(self):
            return {
                'type': 'MultiLineString',
                'coordinates': [
                    [[0.0, 1.0], [1.0, 1.0]],
                    [[1.0, 1.0], [2.0, 2.0]]
                ]
            }

    actual = MultiGeoLineString.from_pyshp(
        MockShape(),
        dt=datetime(2020, 1, 1),
        properties={'test': 'prop'}
    )
    expected = MultiGeoLineString(
        [
            GeoLineString([Coordinate(0., 1., z=10., m=11.), Coordinate(1., 1., z=12., m=13.)]),
            GeoLineString([Coordinate(1., 1., z=14., m=15.), Coordinate(2., 2., z=16., m=17.)])
        ],
        dt=datetime(2020, 1, 1),
        properties={'test': 'prop'}
    )

    assert actual == expected
    assert actual.properties == expected.properties


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

    with pytest.raises(ValueError):
        MultiGeoLineString.from_wkt('test')


def test_multigeolinestring_to_geojson():
    mls = MultiGeoLineString(
        [
            GeoLineString([Coordinate(0.0, 1.0), Coordinate(0.5, 0.5), Coordinate(1.0, 0.0)]),
            GeoLineString([Coordinate(1.0, 1.0), Coordinate(0.5, 0.5), Coordinate(0.0, 0.0)])
        ],
        dt=datetime(2020, 1, 1),
        properties={'test_prop': 'test_value'}
    )
    assert mls.to_geojson(test_kwarg='test_kwarg') == {
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
            "datetime_start": "2020-01-01T00:00:00+00:00",
            "datetime_end": "2020-01-01T00:00:00+00:00",
        },
        'test_kwarg': 'test_kwarg'
    }


def test_multigeolinestring_to_pyshp(pyshp_round_trip):
    original = FeatureCollection(
        [
            MultiGeoLineString(
                [
                    GeoLineString([Coordinate(0.0, 1.0), Coordinate(0.5, 0.5), Coordinate(1.0, 0.0)]),
                    GeoLineString([Coordinate(1.0, 1.0), Coordinate(0.5, 0.5), Coordinate(0.0, 0.0)])
                ]
            )
        ]
    )
    new = pyshp_round_trip(original)
    assert set(original.geoshapes) == set(new.geoshapes)

    original = FeatureCollection(
        [
            MultiGeoLineString(
                [
                    GeoLineString([Coordinate(0.0, 1.0, z=1), Coordinate(0.5, 0.5, z=1), Coordinate(1.0, 0.0, z=1)]),
                    GeoLineString([Coordinate(1.0, 1.0, z=1), Coordinate(0.5, 0.5, z=1), Coordinate(0.0, 0.0, z=1)])
                ]
            )
        ]
    )
    new = pyshp_round_trip(original)
    assert set(original.geoshapes) == set(new.geoshapes)

    original = FeatureCollection(
        [
            MultiGeoLineString(
                [
                    GeoLineString([Coordinate(0.0, 1.0, m=1), Coordinate(0.5, 0.5, m=1), Coordinate(1.0, 0.0, m=1)]),
                    GeoLineString([Coordinate(1.0, 1.0, m=1), Coordinate(0.5, 0.5, m=1), Coordinate(0.0, 0.0, m=1)])
                ]
            )
        ]
    )
    new = pyshp_round_trip(original)
    assert set(original.geoshapes) == set(new.geoshapes)


def test_multigeolinestring_to_shapely():
    smls = shapely.MultiLineString([[[0, 0], [1, 2]], [[4, 4], [5, 6]]])
    mls = MultiGeoLineString([
        GeoLineString([Coordinate(0.0, 0.0), Coordinate(1., 2.)]),
        GeoLineString([Coordinate(4.0, 4.0), Coordinate(5., 6.)])
    ])
    assert mls.to_shapely() == smls


def test_multigeolinestring_to_wkt():
    mls = MultiGeoLineString([
        GeoLineString([Coordinate(0.0, 1.0), Coordinate(0.5, 0.5), Coordinate(1.0, 0.0)]),
        GeoLineString([Coordinate(1.0, 1.0), Coordinate(0.5, 0.5), Coordinate(0.0, 0.0)])
    ])
    assert mls.to_wkt() == "MULTILINESTRING((0.0 1.0,0.5 0.5,1.0 0.0), (1.0 1.0,0.5 0.5,0.0 0.0))"

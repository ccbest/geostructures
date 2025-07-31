from datetime import datetime

import pytest
import shapely

from geostructures import MultiGeoPoint, GeoPoint, Coordinate, GeoCircle, GeoPolygon, FeatureCollection
from geostructures.time import TimeInterval

from tests.functions import pyshp_round_trip


def test_multigeopoint_hash():
    assert len({
        MultiGeoPoint([
            GeoPoint(Coordinate(0., 0.)),
            GeoPoint(Coordinate(1., 1.))
        ]),
        MultiGeoPoint([
            GeoPoint(Coordinate(0., 0.)),
            GeoPoint(Coordinate(1., 1.))
        ])
    }) == 1

    assert len({
        MultiGeoPoint([
            GeoPoint(Coordinate(0., 0.)),
            GeoPoint(Coordinate(1., 1.))
        ]),
        MultiGeoPoint([
            GeoPoint(Coordinate(0., 0.)),
            GeoPoint(Coordinate(2., 1.))
        ])
    }) == 2


def test_multigeopoint_repr():
    mp = MultiGeoPoint([
        GeoPoint(Coordinate(0., 0.)),
        GeoPoint(Coordinate(1., 1.))
    ])
    assert repr(mp) == '<MultiGeoPoint of 2 points>'

    mp = MultiGeoPoint([
        GeoPoint(Coordinate(0., 0.)),
    ])
    assert repr(mp) == '<MultiGeoPoint of 1 point>'


def test_multigeopoint_centroid():
    mp = MultiGeoPoint([
        GeoPoint(Coordinate(0., 0.)),
        GeoPoint(Coordinate(1., 1.))
    ])
    assert mp.centroid == Coordinate(0.5, 0.5)


def test_multigeopoint_circumscribing_circle():
    mp = MultiGeoPoint([
        GeoPoint(Coordinate(0., 0.)),
        GeoPoint(Coordinate(1., 1.))
    ])
    assert mp.circumscribing_circle() == GeoCircle(Coordinate(0.5, 0.5), 78626.18767687456)


def test_multigeopoint_convex_hull():
    mp = MultiGeoPoint([
        GeoPoint(Coordinate(0., 0.)),
        GeoPoint(Coordinate(0., 1.)),
        GeoPoint(Coordinate(1., 1.)),
        GeoPoint(Coordinate(1., 0.)),
    ])
    assert mp.convex_hull() == GeoPolygon([
        Coordinate(0., 0.),
        Coordinate(1., 0.),
        Coordinate(1., 1.),
        Coordinate(0., 1.),
        Coordinate(0., 0.),
    ])


def test_multigeopoint_copy():
    mp = MultiGeoPoint([
        GeoPoint(Coordinate(0., 0.)),
        GeoPoint(Coordinate(1., 1.))
    ])
    mp2 = mp.copy()
    assert mp == mp2

    mp2.geoshapes.pop()
    assert mp != mp2


def test_multigeopoint_from_geojson():
    gjson = {
        "type": "Feature",
        "geometry": {
            "type": "MultiPoint",
            "coordinates": [[0.0, 1.0], [1.0, 1.0]]},
        "properties": {
            "test_prop": "test_value",
            "datetime_start": "2020-01-01T00:00:00+00:00",
            "datetime_end": "2020-01-01T00:00:00+00:00"
        }
    }
    mp = MultiGeoPoint.from_geojson(gjson)
    assert mp.geoshapes == [
        GeoPoint(Coordinate(0., 1.)),
        GeoPoint(Coordinate(1., 1.))
    ]
    assert mp._properties == {"test_prop": "test_value"}
    assert mp.dt == TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 1))

    # Only geo interface
    gjson = gjson['geometry']
    mp = MultiGeoPoint.from_geojson(gjson)
    assert mp.geoshapes == [
        GeoPoint(Coordinate(0., 1.)),
        GeoPoint(Coordinate(1., 1.))
    ]

    with pytest.raises(ValueError):
        gjson = {
            "type": "Feature",
            "geometry": {
                "type": "test",
            }
        }
        MultiGeoPoint.from_geojson(gjson)


def test_multigeopoint_from_shapely():
    shapely_mp = shapely.MultiPoint([(0., 1.), (1., 1.)])
    mp = MultiGeoPoint.from_shapely(shapely_mp)
    assert mp.geoshapes == [
        GeoPoint(Coordinate(0., 1.)),
        GeoPoint(Coordinate(1., 1.))
    ]


def test_multigeopoint_from_wkt():
    wkt = "MULTIPOINT (0 1, 1 1)"
    mp = MultiGeoPoint.from_wkt(wkt)
    assert mp.geoshapes == [
        GeoPoint(Coordinate(0., 1.)),
        GeoPoint(Coordinate(1., 1.))
    ]

    with pytest.raises(ValueError):
        MultiGeoPoint.from_wkt('test')


def test_multigeopoint_to_geojson():
    mp = MultiGeoPoint(
        [
            GeoPoint(Coordinate(0., 1.)),
            GeoPoint(Coordinate(1., 1.))
        ],
        dt=datetime(2020, 1, 1),
        properties={'test_prop': 'test_val'},
    )
    assert mp.to_geojson(test_kwarg="test_kwarg") == {
        "type": "Feature",
        "geometry": {
            "type": "MultiPoint",
            "coordinates": [[0.0, 1.0], [1.0, 1.0]]},
        "properties": {
            "test_prop": "test_val",
            "datetime_start": "2020-01-01T00:00:00+00:00",
            "datetime_end": "2020-01-01T00:00:00+00:00"
        },
        "test_kwarg": "test_kwarg"
    }


def test_multigeopoint_to_pyshp(pyshp_round_trip):

    original = FeatureCollection([
            MultiGeoPoint(
            [GeoPoint(Coordinate(0., 1., z=1., m=3.)), GeoPoint(Coordinate(1., 1., z=2., m=4.))],
            dt=datetime(2020, 1, 1),
            properties={'test': 'prop'}
        )
    ])
    new = pyshp_round_trip(original)
    assert set(original.geoshapes) == set(new.geoshapes)

    original = FeatureCollection([
        MultiGeoPoint(
            [GeoPoint(Coordinate(0., 1., m=3.)), GeoPoint(Coordinate(1., 1., m=4.))],
            dt=datetime(2020, 1, 1),
            properties={'test': 'prop'}
        )
    ])
    new = pyshp_round_trip(original)
    assert set(original.geoshapes) == set(new.geoshapes)


def test_multigeopoint_to_shapely():
    mp = MultiGeoPoint(
        [
            GeoPoint(Coordinate(0., 1.)),
            GeoPoint(Coordinate(1., 1.))
        ],
    )
    assert mp.to_shapely() == shapely.MultiPoint([(0., 1.), (1., 1.)])


def test_multigeopoint_to_wkt():
    mp = MultiGeoPoint([
        GeoPoint(Coordinate(0., 1.)),
        GeoPoint(Coordinate(1., 1.))
    ])
    assert mp.to_wkt() == "MULTIPOINT(0.0 1.0, 1.0 1.0)"

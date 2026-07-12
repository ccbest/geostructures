
from datetime import datetime

import pytest
import shapely

from geostructures import GeoCircle, GeoPoint, MultiGeoPoint
from geostructures.coordinates import Coordinate

from tests.functions import (
    default_test_datetime, geojson_round_trip, shapely_round_trip, wkt_round_trip,
)


def test_geopoint_eq():
    point = GeoPoint(Coordinate('0.0', '0.0'), dt=default_test_datetime)

    p2 = GeoPoint(Coordinate('0.0', '0.0'), dt=default_test_datetime)
    assert point == p2

    p2 = GeoPoint(Coordinate('0.0', '1.0'), dt=default_test_datetime)
    assert point != p2

    p2 = GeoPoint(Coordinate('0.0', '0.0'), dt=datetime(1970, 1, 1, 1, 1))
    assert point != p2

    p2 = 'not a geopoint'
    assert point != p2


def test_geopoint_hash():
    point = GeoPoint(Coordinate('0.0', '0.0'), dt=default_test_datetime)

    p2 = GeoPoint(Coordinate('0.0', '0.0'), dt=default_test_datetime)
    assert len({point, p2}) == 1

    p2 = GeoPoint(Coordinate('0.0', '1.0'), dt=default_test_datetime)
    assert len({point, p2}) == 2


def test_geopoint_repr():
    point = GeoPoint(Coordinate('0.0', '0.0'))
    assert repr(point) == "<GeoPoint at (0.0, 0.0)>"


def test_geopoint_bounds():
    point = GeoPoint(Coordinate(0., 0.))
    assert point.bounds == (0., 0., 0., 0.)


def test_geopoint_centroid():
    point = GeoPoint(Coordinate(0., 0.))
    assert point.centroid == point.coordinate


def test_geopoint_contains_dunder():
    point = GeoPoint(Coordinate(0., 0.))
    assert point in point


def test_geopoint_contains():
    assert GeoPoint(Coordinate(0., 0.)).contains(GeoPoint(Coordinate(0., 0.)))
    assert not GeoPoint(Coordinate(0., 0.)).contains(GeoPoint(Coordinate(1., 1.)))


def test_geopoint_contains_coordinate():
    assert GeoPoint(Coordinate(0., 0.)).contains_coordinate(Coordinate(0., 0.))
    assert not GeoPoint(Coordinate(0., 0.)).contains_coordinate(Coordinate(1., 0.))


def test_geopoint_contains_shape():
    assert GeoPoint(Coordinate(0., 0.)).contains_shape(GeoPoint(Coordinate(0., 0.)))
    assert GeoPoint(Coordinate(0., 0.)).contains_shape(MultiGeoPoint([GeoPoint(Coordinate(0., 0.))]))
    assert not GeoPoint(Coordinate(0., 0.)).contains_shape(GeoCircle(Coordinate(0., 0.), 10))
    assert not GeoPoint(Coordinate(0., 0.)).contains_shape(MultiGeoPoint([GeoPoint(Coordinate(1., 0.))]))


def test_geopoint_intersects_shape():
    point = GeoPoint(Coordinate(0., 0.))
    assert point.intersects_shape(GeoCircle(Coordinate(0., 0.), 500))
    assert point.intersects_shape(point)

    assert not point.intersects_shape(GeoCircle(Coordinate(1., 0.), 500))
    assert not point.intersects_shape(GeoPoint(Coordinate(0.001, 0.001)))


def test_geopoint_copy():
    point = GeoPoint(Coordinate(0., 1.))
    point_copy = point.copy()

    # Assert equality but different pointer
    assert point == point_copy
    assert point is not point_copy


def test_geopoint_serialization_round_trips():
    point = GeoPoint(Coordinate(1.0, 0.0), dt=default_test_datetime)
    wkt_round_trip(point)
    geojson_round_trip(point)
    shapely_round_trip(point)


def test_geopoint_to_wkt():
    point = GeoPoint(Coordinate(0.0, 0.0))
    assert point.to_wkt() == 'POINT(0 0)'

    # Z/M values emit the corresponding WKT dimensionality designator -
    # without one, M-only coordinates are indistinguishable from Z
    assert GeoPoint(Coordinate(0.0, 0.0, z=1.)).to_wkt() == 'POINT Z(0 0 1)'
    assert GeoPoint(Coordinate(0.0, 0.0, m=2.)).to_wkt() == 'POINT M(0 0 2)'
    assert GeoPoint(Coordinate(0.0, 0.0, z=1., m=2.)).to_wkt() == 'POINT ZM(0 0 1 2)'


def test_geopoint_from_wkt():
    wkt_str = 'POINT (1.0 1.0)'
    assert GeoPoint.from_wkt(wkt_str) == GeoPoint(Coordinate(1.0, 1.0))

    wkt_str = 'POINT(1.0 1.0)'
    assert GeoPoint.from_wkt(wkt_str) == GeoPoint(Coordinate(1.0, 1.0))

    wkt_str = 'POINT(-1.0 -1.0)'
    assert GeoPoint.from_wkt(wkt_str) == GeoPoint(Coordinate(-1.0, -1.0))

    with pytest.raises(ValueError):
        _ = GeoPoint.from_wkt('NOT WKT')


def test_geopoint_from_geojson():
    gpoint = {
        'type': 'Feature',
        'geometry': {
            'type': 'Point',
            'coordinates': [1.0, 0.0]
        },
        'properties': {'example': 'prop'}
    }
    expected = GeoPoint(
        Coordinate(1.0, 0.0),
        properties={'example': 'prop'}
    )
    assert GeoPoint.from_geojson(gpoint) == expected

    # Only geo interface
    gpoint = gpoint['geometry']
    expected = GeoPoint(
        Coordinate(1.0, 0.0),
    )
    assert GeoPoint.from_geojson(gpoint) == expected

    with pytest.raises(ValueError):
        bad_gjson = {
            'type': 'Feature',
            'geometry': {
                'type': 'Error',
                'coordinates': [0.0, 0.0],
            },
            'properties': {'example': 'prop'}
        }
        GeoPoint.from_geojson(bad_gjson)


def test_geopoint_to_geojson():
    point = GeoPoint(Coordinate(1., 0.), dt=default_test_datetime)
    # Assert kwargs and properties end up in the right place
    assert point.to_geojson(properties={'test_prop': 2}, test_kwarg=1) == {
        'type': 'Feature',
        'geometry': {
            'type': 'Point',
            'coordinates': [1.0, 0.0],
        },
        'properties': {
            'test_prop': 2,
            'datetime_start': default_test_datetime.isoformat(),
            'datetime_end': default_test_datetime.isoformat(),
        },
        'test_kwarg': 1
    }


def test_geopoint_to_shapely():
    point = GeoPoint(Coordinate(0.0, 0.0))
    assert point.to_shapely() == shapely.Point(0.0, 0.0)

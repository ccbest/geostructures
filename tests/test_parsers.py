
from datetime import datetime
import json

import pytest

from geostructures import *
from geostructures.parsers import *


def test_parse_fastkml():
    from fastkml import Folder
    folder = Folder(name='test folder')
    folder.append(
        GeoPoint(
            Coordinate(1., 0.),
            dt=datetime(2020, 1, 1),
            properties={'test': 'prop'}
        ).to_fastkml_placemark(
            description='test description',
            name='test name'
        )
    )
    parsed = parse_fastkml(folder)
    assert parsed == [
        GeoPoint(
            Coordinate(1., 0.),
            dt=datetime(2020, 1, 1),
        )
    ]
    assert parsed[0]._properties == {
        'test': 'prop',
        'sub_folder_0': 'test folder',
        'name': 'test name',
        'description': 'test description'
    }


def test_parse_geojson():
    # No need to test properties or times - tested elsewhere
    shape = GeoCircle(Coordinate(0., 0.), 500).to_polygon()
    assert parse_geojson(shape.to_geojson()) == shape

    # Stringified geojson
    assert parse_geojson(json.dumps(shape.to_geojson())) == shape

    # Just the geo interface
    assert parse_geojson(shape.to_geojson()['geometry']) == shape


def test_parse_wkt():
    shape = GeoCircle(Coordinate(0., 0.), 500).to_polygon()
    assert parse_wkt(shape.to_wkt()) == shape

    shape = GeoLineString([Coordinate(0., 0.), Coordinate(0., 0.)])
    assert parse_wkt(shape.to_wkt()) == shape

    shape = GeoPoint(Coordinate(1., 0.))
    assert parse_wkt(shape.to_wkt()) == shape

    shape = MultiGeoPolygon([
        GeoCircle(Coordinate(0., 0.), 500).to_polygon(),
        GeoCircle(Coordinate(0., 0.), 500).to_polygon()
    ])
    assert parse_wkt(shape.to_wkt()) == shape

    shape = MultiGeoLineString([
        GeoLineString([Coordinate(0., 0.), Coordinate(0., 0.)]),
        GeoLineString([Coordinate(0., 0.), Coordinate(0., 0.)])
    ])
    assert parse_wkt(shape.to_wkt()) == shape

    shape = MultiGeoPoint([
        GeoPoint(Coordinate(1., 0.)),
        GeoPoint(Coordinate(1., 0.))
    ])
    assert parse_wkt(shape.to_wkt()) == shape

    shape = GeoPoint(Coordinate(1., 0., z=50., m=10.))
    assert parse_wkt(shape.to_wkt()) == shape

    with pytest.raises(ValueError):
        parse_wkt('123')

    with pytest.raises(ValueError):
        parse_wkt('worse')

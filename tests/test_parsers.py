
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

    with pytest.raises(ValueError):
        parse_wkt('bad wkt')


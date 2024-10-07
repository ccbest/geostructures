
import pytest

from geostructures import *
from geostructures.parsers import parse_wkt, parse_geojson


def test_parse_geojson():
    # No need to test properties or times - tested elsewhere
    shape = GeoCircle(Coordinate(0., 0.), 500).to_polygon()
    assert parse_geojson(shape.to_geojson()) == shape


def test_parse_wkt():
    shape = GeoCircle(Coordinate(0., 0.), 500).to_polygon()
    assert parse_wkt(shape.to_wkt()) == shape

    with pytest.raises(ValueError):
        parse_wkt('bad wkt')


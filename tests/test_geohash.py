
import pytest
from datetime import datetime

from geostructures import Coordinate, GeoBox, GeoCircle, GeoLineString, GeoPoint
from geostructures.collections import FeatureCollection
from geostructures.geohash import (
    _coord_to_niemeyer, _get_niemeyer_subhashes, niemeyer_to_geobox,
    H3Hasher, NiemeyerHasher
)
from geostructures.time import TimeInterval
from geostructures.utils.agg_functions import *


def test_coord_to_niemeyer():
    coord = Coordinate(0.1, -0.1)
    assert _coord_to_niemeyer(coord, 8, 16) == '95555659'

    with pytest.raises(ValueError):
        _ = _coord_to_niemeyer(coord, 8, 42)


def test_get_niemeyer_subhashes():
    geohash = '95555659'
    assert _get_niemeyer_subhashes(geohash, 16) == {
        '955556590', '955556591', '955556592', '955556593', '955556594',
        '955556595', '955556596', '955556597', '955556598', '955556599',
        '95555659a', '95555659b', '95555659c', '95555659d', '95555659e',
        '95555659f',
    }

    with pytest.raises(ValueError):
        _get_niemeyer_subhashes(geohash, 42)


def test_niemeyer_to_geobox():
    geohash = '95555659'
    assert niemeyer_to_geobox(geohash, 16) == GeoBox(
        Coordinate(0.098876953125, -0.098876953125),
        Coordinate(0.1043701171875, -0.1043701171875)
    )

    with pytest.raises(ValueError):
        # Character not in base charset
        _ = niemeyer_to_geobox('95555659z', 16)


def test_h3hasher_hash_coordinates():
    coords = [
        Coordinate(0.0, 0.0), Coordinate(1.0, 1.0),
        Coordinate(0.0, 0.0)
    ]
    hasher = H3Hasher(resolution=8)

    assert hasher.hash_coordinates(coords) == {
        '88754e6499fffff': 2, '887541ad5bfffff': 1
    }


def test_h3hasher_hash_shape():
    shape = GeoCircle(Coordinate(0.0, 0.0), 500)
    hasher_9 = H3Hasher(resolution=9)
    hasher_8 = H3Hasher(resolution=8)

    assert hasher_9.hash_shape(shape) == {
        '89754a9325bffff',
        '89754e64983ffff',
        '89754e64993ffff',
        '89754e64997ffff',
        '89754e6499bffff',
        '89754e64d23ffff',
        '89754e64d27ffff',
        '89754e64d2bffff',
        '89754e64d2fffff',
        '89754e64d67ffff'
    }

    assert hasher_8.hash_shape(shape) == {
        '88754e6499fffff', '88754e64d3fffff'
    }

    shape = GeoLineString(
        [Coordinate(0.0, 0.0), Coordinate(0.01, 0.01), Coordinate(0.02, 0.0)]
    )
    assert hasher_8.hash_shape(shape) == {
        '88754a9363fffff',
        '88754a9367fffff',
        '88754e6499fffff',
        '88754e64d3fffff',
        '88754e64dbfffff'
    }

    shape = GeoPoint(Coordinate(0.0, 0.0))
    assert hasher_8.hash_shape(shape) == {'88754e6499fffff'}

    with pytest.raises(ValueError):
        hasher = H3Hasher()
        hasher.hash_shape(shape)


def test_hash_collection():
    shape = GeoCircle(Coordinate(0.0, 0.0), 600)
    shape2 = GeoCircle(Coordinate(0.0, 0.0), 300)
    fcol = FeatureCollection([shape, shape2])
    hasher = H3Hasher(resolution=9)

    assert hasher.hash_collection(fcol) == {
        '89754e64d2fffff': 2,
        '89754e64d2bffff': 1,
        '89754e64983ffff': 1,
        '89754e64987ffff': 1,
        '89754e64993ffff': 2,
        '89754e64997ffff': 2,
        '89754e64d27ffff': 1,
        '89754e64d67ffff': 1,
        '89754a9324bffff': 1,
        '89754e64d23ffff': 1,
        '89754a9325bffff': 1,
        '89754e6499bffff': 1
    }

    with pytest.raises(ValueError):
        hasher = H3Hasher()
        hasher.hash_collection(fcol)


def test_hash_collection_with_total_time():
    shape = GeoCircle(Coordinate(0.0, 0.0), 600, 
                      dt=TimeInterval(datetime(2024,1,1), datetime(2024,1,1,1)))
    shape2 = GeoCircle(Coordinate(0.0, 0.0), 300,
                      dt=TimeInterval(datetime(2024,1,1), datetime(2024,1,1,2)))
    fcol = FeatureCollection([shape, shape2])
    hasher = H3Hasher(resolution=9)

    assert hasher.hash_collection(fcol, agg_fn=total_time) == {
        '89754e64d2fffff': 10800.0,
        '89754e64d2bffff': 3600.0,
        '89754e64983ffff': 3600.0,
        '89754e64987ffff': 3600.0,
        '89754e64993ffff': 10800.0,
        '89754e64997ffff': 10800.0,
        '89754e64d27ffff': 3600.0,
        '89754e64d67ffff': 3600.0,
        '89754a9324bffff': 3600.0,
        '89754e64d23ffff': 3600.0,
        '89754a9325bffff': 3600.0,
        '89754e6499bffff': 3600.0
    }

def test_hash_collection_with_unique_entities():
    shape = GeoCircle(Coordinate(0.0, 0.0), 600, 
                      properties={'entity': 1})
    shape2 = GeoCircle(Coordinate(0.0, 0.0), 300,
                      properties={'entity': 2})
    shape3 = GeoCircle(Coordinate(0.0, 0.0), 450,
                      properties={'entity': 1})
    fcol = FeatureCollection([shape, shape2, shape3])
    hasher = H3Hasher(resolution=9)

    assert hasher.hash_collection(fcol, agg_fn=unique_entities) == {
        '89754e64d2fffff': 2,
        '89754e64d2bffff': 1,
        '89754e64983ffff': 1,
        '89754e64987ffff': 1,
        '89754e64993ffff': 2,
        '89754e64997ffff': 2,
        '89754e64d27ffff': 1,
        '89754e64d67ffff': 1,
        '89754a9324bffff': 1,
        '89754e64d23ffff': 1,
        '89754a9325bffff': 1,
        '89754e6499bffff': 1
    }

def test_niemeyer_hash_collection():
    hasher = NiemeyerHasher(8, 16)
    col = FeatureCollection([
        GeoCircle(Coordinate(0.0, 0.0), 700),
        GeoPoint(Coordinate(0.0, 0.0)),
        GeoLineString([Coordinate(0.0, 0.0), Coordinate(0.02, 0.03), Coordinate(0.04, 0.0)])
    ])
    assert hasher.hash_collection(col) == {
        '95555554': 1,
        '6aaaaaa8': 1,
        '3fffffff': 3,
        '95555557': 1,
        '3ffffffd': 1,
        'c0000000': 2,
        'c0000001': 2,
        '95555555': 2,
        '3ffffffe': 1,
        '6aaaaaaa': 2,
        'c0000002': 1,
        '6aaaaaab': 1,
        'c0000003': 1,
        'c0000007': 1,
        'c000000c': 1,
        'c000002a': 1,
        'c0000029': 1,
        'c000000d': 1,
        '9555557f': 1,
        'c0000006': 1,
        'c0000018': 1,
        'c000001a': 1,
        'c0000025': 1,
        'c0000023': 1,
        'c0000028': 1,
        'c0000026': 1,
        'c0000030': 1,
        'c0000027': 1,
        'c000001b': 1
     }


def test_niemeyer_hash_coordinates():
    coords = [
        Coordinate(0.0, 0.0), Coordinate(1.0, 1.0),
        Coordinate(0.0, 0.0)
    ]
    hasher = NiemeyerHasher(8, 16)

    assert hasher.hash_coordinates(coords) == {
        '3fffffff': 2, 'c000cf3c': 1
    }


def test_niemeyer_hash_shape():
    hasher = NiemeyerHasher(8, 16)

    shape = GeoCircle(Coordinate(0.0, 0.0), 700)
    assert hasher.hash_shape(shape) == {
        '3ffffffd', '3ffffffe', '3fffffff', '6aaaaaa8', '6aaaaaaa', '6aaaaaab',
        '95555554', '95555555', '95555557', 'c0000000', 'c0000001', 'c0000002'
    }

    shape = GeoPoint(Coordinate(0.0, 0.0))
    assert hasher.hash_shape(shape) == {'3fffffff'}

    shape = GeoLineString([Coordinate(0.0, 0.0), Coordinate(0.02, 0.03), Coordinate(0.04, 0.0)])
    assert hasher.hash_shape(shape) == {
        '3fffffff', '6aaaaaaa', '95555555', '9555557f', 'c0000000', 'c0000001',
        'c0000003', 'c0000006', 'c0000007', 'c000000c', 'c000000d', 'c0000018',
        'c000001a', 'c000001b', 'c0000023', 'c0000025', 'c0000026', 'c0000027',
        'c0000028', 'c0000029', 'c000002a', 'c0000030'
    }

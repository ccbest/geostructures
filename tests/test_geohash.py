
import pytest

from geostructures import Coordinate, GeoCircle, GeoLineString, GeoPoint
from geostructures.collections import FeatureCollection
from geostructures.geohash import H3Hasher


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

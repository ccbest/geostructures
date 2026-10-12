import pytest
from datetime import datetime
import math

from geostructures import *
from geostructures._geometry import gaussian_probability_in_polygons
from geostructures.collections import FeatureCollection
from geostructures.geohash import (
    _coord_to_niemeyer, _get_niemeyer_subhashes, h3_to_geopolygon, niemeyer_to_geobox,
    H3Hasher, HasherBase, NiemeyerHasher
)
from geostructures.time import TimeInterval
from geostructures.utils.agg_functions import *

from tests.functions import assert_shape_equivalence


def test_coord_to_niemeyer():
    coord = Coordinate(0.1, -0.1)
    assert _coord_to_niemeyer(coord, 8, 16) == '9555534d'

    with pytest.raises(ValueError):
        _ = _coord_to_niemeyer(coord, 8, 42)


def test_get_niemeyer_subhashes():
    geohash = '95555659'
    assert _get_niemeyer_subhashes(geohash, 16) == {
        '955556590', '955556591', '955556592', '955556593', '955556594', '955556595',
        '955556596', '955556597', '955556598', '955556599', '95555659a', '95555659b',
        '95555659c', '95555659d', '95555659e', '95555659f'
    }

    with pytest.raises(ValueError):
        _get_niemeyer_subhashes(geohash, 42)


def test_niemeyer_to_geobox():
    geohash = '95555659'
    assert niemeyer_to_geobox(geohash, 16) == GeoBox(
        Coordinate(0.098876953125, -0.0494384765625),
        Coordinate(0.1043701171875, -0.05218505859375)
    )

    with pytest.raises(ValueError):
        # Character not in base charset
        _ = niemeyer_to_geobox('95555659z', 16)


def test_hasherbase_unsupported():
    class _Hasher(HasherBase):
        def hash_collection(self, collection, **kwargs):
            return {}

        def hash_shape(self, shape, **kwargs):
            return set()

    hasher = _Hasher()
    with pytest.raises(NotImplementedError):
        hasher.coarsen()

    with pytest.raises(NotImplementedError):
        hasher.get_cell_boundary('a')

    with pytest.raises(NotImplementedError):
        hasher.get_neighbors('a')


def test_h3hasher_coarsen():
    assert H3Hasher(5).coarsen().resolution == 4
    assert H3Hasher(1).coarsen().resolution == 0
    assert H3Hasher(0).coarsen() is None

    with pytest.raises(ValueError):
        H3Hasher().coarsen()

    # Resolution 0 (the base cells) is valid
    assert H3Hasher(0).hash_shape(GeoPoint(Coordinate(0., 0.))) == {'8075fffffffffff'}


def test_h3hasher_get_cell_boundary():
    assert H3Hasher(7).get_cell_boundary('87195da49ffffff') == h3_to_geopolygon('87195da49ffffff').bounding_coords()


def test_h3hasher_get_neighbors():
    import h3

    actual = H3Hasher(7).get_neighbors('87195da49ffffff')
    assert len(actual) == 6
    assert '87195da49ffffff' not in actual
    assert all(h3.are_neighbor_cells(x, '87195da49ffffff') for x in actual)


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

    # Regression: Z/M-carrying polygon coordinates previously crashed h3
    # (h3.LatLngPoly only accepts 2D points)
    zm_shape = GeoPolygon([
        Coordinate(0., 0., z=1., m=2.), Coordinate(0.01, 0., z=1.),
        Coordinate(0.01, 0.01, z=1.), Coordinate(0., 0., z=1., m=2.),
    ])
    assert hasher_8.hash_shape(zm_shape)

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
                      dt=TimeInterval(datetime(2024, 1, 1), datetime(2024, 1, 1, 1)))
    shape2 = GeoCircle(Coordinate(0.0, 0.0), 300,
                       dt=TimeInterval(datetime(2024, 1, 1), datetime(2024, 1, 1, 2)))
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

def test_niemeyer_coarsen():
    actual = NiemeyerHasher(5, 32).coarsen()
    assert actual.length == 4
    assert actual.base == 32
    assert NiemeyerHasher(1, 32).coarsen() is None


def test_niemeyer_get_cell_boundary():
    assert NiemeyerHasher(4, 32).get_cell_boundary('9q8y') == niemeyer_to_geobox('9q8y', 32).bounding_coords()


def test_niemeyer_get_neighbors():
    actual = NiemeyerHasher(4, 32).get_neighbors('9q8y')
    assert len(actual) == 8
    assert actual == set(NiemeyerHasher._get_surrounding('9q8y', 32))


def test_niemeyer_hash_collection():
    hasher = NiemeyerHasher(8, 16)
    col = FeatureCollection([
        GeoCircle(Coordinate(0.0, 0.0), 700),
        GeoPoint(Coordinate(0.0, 0.0)),
        GeoLineString([Coordinate(0.0, 0.0), Coordinate(0.02, 0.03), Coordinate(0.04, 0.0)])
    ])
    assert hasher.hash_collection(col) == {
        'c0000000': 2,
        '6aaaaaa8': 1,
        '6aaaaaae': 1,
        'c0000004': 2,
        '6aaaaaab': 1,
        '3fffffff': 3,
        'c0000001': 2,
        '3ffffffe': 1,
        'c0000003': 1,
        '95555556': 1,
        '95555554': 1,
        'c0000002': 1,
        '3ffffffd': 1,
        '3ffffffc': 1,
        '95555551': 1,
        '6aaaaaaa': 2,
        '3ffffffb': 1,
        '6aaaaaa9': 1,
        '95555555': 2,
        '95555557': 1,
        'c0000033': 1,
        'c000001c': 1,
        'c000004b': 1,
        'c0000013': 1,
        'c0000016': 1,
        'c0000027': 1,
        'c0000019': 1,
        'c0000060': 1,
        'c0000029': 1,
        'c0000048': 1,
        'c0000006': 1,
        'c0000012': 1,
        'c0000032': 1,
        'c0000061': 1,
        'c0000007': 1,
        'c000002d': 1,
        '9555557f': 1,
        'c000002c': 1,
        'c0000005': 1,
        'c000004a': 1,
        'c0000028': 1,
        'c0000049': 1,
        'c000002a': 1,
        'c0000036': 1,
        'c000004e': 1,
        'c0000035': 1,
        'c0000034': 1,
        'c000001d': 1
    }


def test_niemeyer_hash_coordinates():
    coords = [
        Coordinate(0.0, 0.0), Coordinate(1.0, 1.0),
        Coordinate(0.0, 0.0)
    ]
    hasher = NiemeyerHasher(8, 16)

    assert hasher.hash_coordinates(coords) == {
        '3fffffff': 2, 'c0019e78': 1
    }


def test_niemeyer_hash_shape():
    hasher = NiemeyerHasher(8, 16)

    shape = GeoCircle(Coordinate(0.0, 0.0), 700)
    assert hasher.hash_shape(shape) == {
        '3ffffffb', '3ffffffc', '3ffffffd', '3ffffffe', '3fffffff', '6aaaaaa8', '6aaaaaa9',
        '6aaaaaaa', '6aaaaaab', '6aaaaaae', '95555551', '95555554', '95555555', '95555556',
        '95555557', 'c0000000', 'c0000001', 'c0000002', 'c0000003', 'c0000004'
    }
    shape = MultiGeoPolygon([
        GeoCircle(Coordinate(0.0001, 0.0001), 5),
        GeoCircle(Coordinate(1.0001, 1.0001), 5)
    ])
    assert hasher.hash_shape(shape) == {'c0000000', 'c0019e78'}

    shape = GeoPoint(Coordinate(0.0, 0.0))
    assert hasher.hash_shape(shape) == {'3fffffff'}
    shape = MultiGeoPoint([
        GeoPoint(Coordinate(0.0, 0.0)),
        GeoPoint(Coordinate(1.0, 1.0))
    ])
    assert hasher.hash_shape(shape) == {'3fffffff', 'c0019e78'}

    shape = GeoLineString([Coordinate(0.0, 0.0), Coordinate(0.001, 0.001)])
    assert hasher.hash_shape(shape) == {'3fffffff', '6aaaaaaa', '95555555', 'c0000000'}

    shape = MultiGeoLineString([
        GeoLineString([Coordinate(0.0, 0.0), Coordinate(0.001, 0.001)]),
        GeoLineString([Coordinate(0.1, 0.1), Coordinate(0.1001, 0.1001)]),
    ])
    assert hasher.hash_shape(shape) == {'3fffffff', '6aaaaaaa', '95555555', 'c0000000', 'c0000618'}


def test_h3_to_geopolygon():
    expected = GeoPolygon(
        [
            Coordinate(-0.14556, 51.52194),
            Coordinate(-0.1602, 51.51508),
            Coordinate(-0.15716, 51.50285),
            Coordinate(-0.13948, 51.49748),
            Coordinate(-0.12484, 51.50435),
            Coordinate(-0.12788, 51.51658),
            Coordinate(-0.14556, 51.52194)
        ],
        dt=datetime(2020, 1, 1),
        properties={
            'h3_geohash': '87195da49ffffff',
            'test': 'prop',
        }
    )
    actual = h3_to_geopolygon('87195da49ffffff', dt=datetime(2020, 1, 1), properties={'test': 'prop'})
    assert_shape_equivalence(actual, expected, precision=5)
    assert actual.properties == expected.properties


def test_h3_hash_shape_multishape_resolution():
    import h3

    multi = MultiGeoPoint([
        GeoPoint(Coordinate(0., 0.)),
        GeoPoint(Coordinate(0.01, 0.01)),
    ])

    # The caller's resolution must reach the recursion into sub-shapes, even
    # when the hasher was constructed without a default
    hasher = H3Hasher()
    hashes = hasher.hash_shape(multi, resolution=9)
    assert hashes
    assert all(h3.get_resolution(h) == 9 for h in hashes)

    # An explicit resolution overrides the hasher default
    hasher = H3Hasher(resolution=7)
    assert hasher.hash_shape(multi, resolution=9) == hashes


def test_h3_hasher_requires_resolution():
    with pytest.raises(ValueError):
        H3Hasher().hash_coordinates([Coordinate(0., 0.)])


def test_hash_gaussian():
    import h3

    ellipse = GeoEllipse(Coordinate(-77.03, 38.90), 2000, 800, 30)

    # Achievable at the hasher's resolution
    hasher = NiemeyerHasher(6, 32)
    expected = hasher.hash_gaussian(ellipse, 0.95, 0.9, 50)
    assert 1 < len(expected) <= 50
    assert all(len(x) == 6 for x in expected)
    assert sum(expected.values()) >= 0.9
    # The least probable hash was required to reach the mass
    assert sum(expected.values()) - min(expected.values()) < 0.9

    # Hashes are selected most probable first
    k = math.sqrt(-2 * math.log(1 - 0.95))
    neighbors = [y for x in expected for y in hasher.get_neighbors(x) if y not in expected]
    neighbor_probs = gaussian_probability_in_polygons(
        [hasher.get_cell_boundary(x) for x in neighbors],
        ellipse.center, 2000 / k, 800 / k, 30
    )
    assert min(expected.values()) >= max(neighbor_probs)

    # Requires coarsening
    assert NiemeyerHasher(12, 32).hash_gaussian(ellipse, 0.95, 0.9, 50) == expected

    actual = H3Hasher(15).hash_gaussian(ellipse, 0.95, 0.9, 50)
    assert len(actual) <= 50
    assert {h3.get_resolution(x) for x in actual} == {9}
    assert sum(actual.values()) >= 0.9

    actual = H3Hasher(15).hash_gaussian(ellipse, 0.95, 0.9, 1)
    assert len(actual) == 1
    assert sum(actual.values()) >= 0.9

    # A larger contour means a tighter distribution, so fewer hashes are required
    actual = hasher.hash_gaussian(ellipse, 0.99, 0.9, 50)
    assert len(actual) < len(expected)

    # Centered on the corner of the four coarsest geohashes, so coarsens past them
    assert NiemeyerHasher(3, 32).hash_gaussian(
        GeoEllipse(Coordinate(0., 0.), 2000, 800, 30), 0.95, 0.9, 1
    ) == {'': 1.}

    # Centered on a corner of resolution 1 cells, but within a base cell
    lat, lon = h3.vertex_to_latlng(h3.cell_to_vertexes(h3.latlng_to_cell(0., 0., 1))[0])
    actual = H3Hasher(3).hash_gaussian(GeoEllipse(Coordinate(lon, lat), 2000, 800, 30), 0.95, 0.9, 1)
    assert list(actual) == ['8083fffffffffff']
    assert actual['8083fffffffffff'] == pytest.approx(1.)

    # Centered on a corner of base cells
    lat, lon = h3.vertex_to_latlng(h3.cell_to_vertexes(h3.latlng_to_cell(0., 0., 0))[0])
    assert H3Hasher(3).hash_gaussian(
        GeoEllipse(Coordinate(lon, lat), 2000, 800, 30), 0.95, 0.9, 1
    ) == {'': 1.}

    for contour, mass, max_hashes in ((0., 0.9, 1), (1., 0.9, 1), (0.95, 0., 1), (0.95, 1., 1), (0.95, 0.9, 0)):
        with pytest.raises(ValueError):
            hasher.hash_gaussian(ellipse, contour, mass, max_hashes)

    with pytest.raises(TypeError):
        hasher.hash_gaussian(GeoBox(Coordinate(0., 1.), Coordinate(1., 0.)), 0.95, 0.9, 1)


def test_hash_gaussian_off_center():
    # A thin ellipse along the NW-SE diagonal through a corner shared by four geohashes,
    # centered just inside the SW geohash. The NW and SE geohashes each hold nearly half
    # the probability, while the SW geohash containing the center holds very little.
    hasher = NiemeyerHasher(6, 32)
    corner_lon, corner_lat = niemeyer_to_geobox('dqcjr1', 32).bounds[2:]
    offset = 1 / 111_000  # about one meter
    ellipse = GeoEllipse(Coordinate(corner_lon - offset, corner_lat - offset), 400, 15, 135)

    # The two best geohashes suffice, without spending a slot on the center geohash
    actual = hasher.hash_gaussian(ellipse, 0.95, 0.95, 2)
    assert set(actual) == {'dqcjr3', 'dqcjr4'}
    assert sum(actual.values()) == pytest.approx(0.9756, abs=1e-4)
    assert hasher.hash_gaussian(ellipse, 0.95, 0.95, 10) == actual

    # The center geohash is included only once its probability is needed
    actual = hasher.hash_gaussian(ellipse, 0.95, 0.99, 10)
    assert set(actual) == {'dqcjr1', 'dqcjr3', 'dqcjr4'}


def test_hash_gaussian_circle():
    # A circle is a Gaussian with equal axes, whatever the rotation
    circle = GeoCircle(Coordinate(-77.03, 38.90), 2000)
    for _hasher in (NiemeyerHasher(12, 32), H3Hasher(15)):
        actual = _hasher.hash_gaussian(circle, 0.95, 0.9, 50)
        expected = _hasher.hash_gaussian(GeoEllipse(Coordinate(-77.03, 38.90), 2000, 2000, 45), 0.95, 0.9, 50)
        assert actual.keys() == expected.keys()
        assert list(actual.values()) == pytest.approx(list(expected.values()))

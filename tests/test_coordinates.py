
import pytest

from geostructures import Coordinate


def test_coordinate_init():
    c = Coordinate(0., 1.)
    assert c.longitude == 0.
    assert c.latitude == 1.

    c = Coordinate('0.0', '1.0')
    assert c.longitude == 0.
    assert c.latitude == 1.

    # Test longitude adjustment
    assert Coordinate(181., 0) == Coordinate(-179., 0)
    assert Coordinate(361., 0) == Coordinate(1., 0.)
    assert Coordinate(-181, 0) == Coordinate(179, 0)
    assert Coordinate(-361, 0) == Coordinate(-1, 0)

    # Test latitude adjustment
    assert Coordinate(1, 91) == Coordinate(-179, 89)
    assert Coordinate(1, 271) == Coordinate(1, -89)
    assert Coordinate(1, -91) == Coordinate(-179, -89)
    assert Coordinate(1, -271) == Coordinate(1, 89)

    # Test unbounded coordinates don't auto-adjust
    assert Coordinate(360, 180, _bounded=False).to_float() == (360, 180)


def test_coordinate_hash():
    coords = [
        Coordinate(0., 0.),
        Coordinate(0., 0.),
        Coordinate(1., 1.)
    ]
    assert len(set(coords)) == 2
    assert Coordinate(0., 0.) in set(coords)
    assert Coordinate(1., 1.) in set(coords)


def test_coordinate_eq():
    assert Coordinate(0., 0.) == Coordinate(0., 0.)
    assert Coordinate(0., 0.) != Coordinate(1., 0.)
    assert Coordinate(0., 0.) != (0., 0.)


def test_coordinate_repr():
    assert repr(Coordinate(0., 1.)) == '<Coordinate(0.0, 1.0)>'


def test_coordinate_to_float():
    assert Coordinate(0., 1.).to_float() == (0.0, 1.0)


def test_coordinate_to_position():
    # Positions carry Z (altitude) but never M - position-based formats
    # (GeoJSON, KML, shapely) define no third slot for M
    assert Coordinate(0., 1.).to_position() == [0.0, 1.0]
    assert Coordinate(0., 1., z=5.).to_position() == [0.0, 1.0, 5.0]
    assert Coordinate(0., 1., m=9.).to_position() == [0.0, 1.0]
    assert Coordinate(0., 1., z=5., m=9.).to_position() == [0.0, 1.0, 5.0]


def test_coordinate_to_str():
    assert Coordinate(0., 1.).to_str() == ('0', '1')
    assert Coordinate(0.1, 1.1).to_str() == ('0.1', '1.1')


def test_coordinate_to_mgrs():
    assert Coordinate(0., 0.).to_mgrs() == '31NAA6602100000'


def test_coordinate_from_mgrs():
    assert [round(x, 5) for x in Coordinate.from_mgrs('31NAA6602100000').to_float()] == [0., 0.]

def test_coordinate_from_projection():
    assert Coordinate.from_projection(2000,3000,'EPSG:3857') == Coordinate(0.017966, 0.026949)

def test_coordinate_to_projection():
    assert Coordinate(0.017966, 0.026949).to_projection('EPSG:3857') == Coordinate(1999.965972, 2999.949068, _bounded=False)


def test_coordinate_to_dms():
    assert Coordinate(-0.118092, 51.509865).to_dms() == ((0, 7, 5.1312, 'W'), (51, 30, 35.514, 'N'))


def test_coordinate_from_dms():
    assert Coordinate.from_dms((0, 0, 0.0, 'E'), (0, 0, 0.0, 'N')) == Coordinate(0., 0.)
    assert Coordinate.from_dms((0, 7, 5.1312, 'W'), (51, 30, 35.514, 'N')) == Coordinate(-0.118092, 51.509865)


def test_coordinate_to_qdms():
    assert Coordinate(-0.118092, 51.509865).to_qdms() == ('W000070513', 'N51303551')


def test_coordinate_from_qdms():
    assert Coordinate.from_qdms('W000070513', 'N51303551') == Coordinate(-0.118092, 51.509864)


def test_coordinate_unbounded_preserves_180():
    # Bounded coordinates normalize longitude 180 to -180
    assert Coordinate(180., 10.).longitude == -180.

    # Unbounded coordinates (used for antimeridian-crossing edge math)
    # must not be normalized
    assert Coordinate(180., 10., _bounded=False).longitude == 180.
    assert Coordinate(185., 10., _bounded=False).longitude == 185.



def test_coordinate_to_qdms_reverse():
    lat_str, lon_str = Coordinate(-0.118092, 51.509865).to_qdms(reverse=True)
    assert (lon_str, lat_str) == ('W000070513', 'N51303551')


def test_coordinate_to_str_with_z_m():
    assert Coordinate(0.5, 1.5, z=2.).to_str() == ('0.5', '1.5', '2')
    assert Coordinate(0.5, 1.5, z=2., m=3.).to_str() == ('0.5', '1.5', '2', '3')
    assert Coordinate(0.5, 1.5).to_str(reverse=True) == ('1.5', '0.5')


def test_coordinate_immutable():
    import pickle

    coord = Coordinate(0., 1., z=2.)
    with pytest.raises(AttributeError):
        coord.longitude = 5.
    with pytest.raises(AttributeError):
        del coord.latitude
    with pytest.raises(AttributeError):
        coord.new_attribute = 'nope'

    # Hash remains stable and pickling works
    assert hash(coord) == hash(Coordinate(0., 1., z=2.))
    assert pickle.loads(pickle.dumps(coord)) == coord

    # The xyz cache does not affect equality or mutability
    _ = coord.xyz
    assert coord.xyz == Coordinate(0., 1., z=2.).xyz


def test_coordinate_type_coercion():
    # Strings, ints, and numpy scalars all coerce to float
    import numpy as np

    coord = Coordinate('0.5', 1, z='2', m=np.float64(3.))
    assert coord.longitude == 0.5
    assert coord.latitude == 1.
    assert coord.z == 2.
    assert coord.m == 3.
    assert all(isinstance(v, float) for v in (coord.longitude, coord.latitude, coord.z, coord.m))

    with pytest.raises(ValueError):
        Coordinate('not a number', 0.)

    with pytest.raises(TypeError):
        Coordinate(None, 0.)

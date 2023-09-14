
import pytest

from geostructures import Coordinate
from geostructures.coordinates import Latitude, Longitude


def test_coordinate_init():
    c = Coordinate(0., 0.)
    assert c.longitude == Longitude(0.)
    assert c.latitude == Latitude(0.)

    c = Coordinate('0.0', '0.0')
    assert c.longitude == Longitude(0.)
    assert c.latitude == Latitude(0.)

    c = Coordinate('0.0', '0.001', same_precision=False)
    assert c.longitude == Longitude(0.001)
    assert c.latitude == Latitude(0.)

    with pytest.raises(ValueError):
        _ = Coordinate('0.', '0.001', same_precision=False, precision=2)


def test_coordinate_hash():
    coords = [
        Coordinate(0., 0.),
        Coordinate(0., 0.),
        Coordinate(1., 1.)
    ]
    assert len(set(coords)) == 2
    assert Coordinate(0., 0.) in set(coords)
    assert Coordinate(1., 1.) in set(coords)


def test_coordinate_repr():
    assert repr(Coordinate(0., 1.)) == '<Coordinate(0.0, 1.0)>'


def test_coordinate_precision():
    assert Coordinate('1.0', '1.0').precision == 1
    assert Coordinate('1.000', '1.0', same_precision=False).precision == 1
    assert Coordinate('1.000', '1.0').precision == 3
    assert Coordinate('1.000', '1.000').precision == 3


def test_coordinate_to_float():
    assert Coordinate(0., 1.).to_float() == (0.0, 1.0)


def test_coordinate_to_str():
    assert Coordinate(0., 1.).to_str() == ('0.0', '1.0')


def test_coordinate_to_mgrs():
    assert Coordinate(0., 0.).to_mgrs() == '31NAA6602100000'


def test_coordinate_from_mgrs():
    assert Coordinate.from_mgrs('31NAA6602100000') == Coordinate(0., 0.)


def test_coordinate_to_dms():
    assert Coordinate(0., 0.).to_dms() == ((0, 0, 0.0), (0, 0, 0.0))


def test_coordinate_from_dms():
    assert Coordinate.from_dms((0, 0, 0.0), (0, 0, 0.0)) == Coordinate(0., 0.)


def test_coordinate_to_qdms():
    assert Coordinate(0., 0.).to_qdms() == ('N000000000', 'E00000000')


def test_coordinate_from_qdms():
    assert Coordinate.from_qdms('N000000000', 'E00000000') == Coordinate(0., 0.)


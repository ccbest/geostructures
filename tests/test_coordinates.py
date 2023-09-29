
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


def test_coordinate_to_str():
    assert Coordinate(0., 1.).to_str() == ('0.0', '1.0')


def test_coordinate_to_mgrs():
    assert Coordinate(0., 0.).to_mgrs() == '31NAA6602100000'


def test_coordinate_from_mgrs():
    assert [round(x, 5) for x in Coordinate.from_mgrs('31NAA6602100000').to_float()] == [0., 0.]


def test_coordinate_to_dms():
    assert Coordinate(-0.118092, 51.509865).to_dms() == ((0, 7, 5.1312, 'W'), (51, 30, 35.514, 'N'))


def test_coordinate_from_dms():
    assert Coordinate.from_dms((0, 0, 0.0, 'E'), (0, 0, 0.0, 'N')) == Coordinate(0., 0.)
    assert Coordinate.from_dms((0, 7, 5.1312, 'W'), (51, 30, 35.514, 'N')) == Coordinate(-0.118092, 51.509865)


def test_coordinate_to_qdms():
    assert Coordinate(-0.118092, 51.509865).to_qdms() == ('W000070513', 'N51303551')


def test_coordinate_from_qdms():
    assert Coordinate.from_qdms('W000070513', 'N51303551') == Coordinate(-0.118092, 51.509864)


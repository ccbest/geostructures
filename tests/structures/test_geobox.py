
from datetime import datetime

import pytest
import shapely

from geostructures import GeoBox, GeoCircle, GeoPolygon
from geostructures.coordinates import Coordinate

from tests.functions import (
    assert_shape_equivalence, default_test_datetime,
    geojson_round_trip, shapely_round_trip, wkt_round_trip,
)


def test_geobox_corner_validation():
    with pytest.raises(ValueError):
        # NW corner east of SE corner (antimeridian-spanning)
        GeoBox(Coordinate(170., 10.), Coordinate(-170., -10.))

    with pytest.raises(ValueError):
        # NW corner south of SE corner
        GeoBox(Coordinate(0., 0.), Coordinate(1., 1.))


def test_geobox_eq():
    box = GeoBox(Coordinate(0.0, 1.0), Coordinate(1.0, 0.0), dt=default_test_datetime)
    b2 = GeoBox(Coordinate(0.0, 1.0), Coordinate(1.0, 0.0), dt=default_test_datetime)
    assert b2 == box

    # Different vertices
    b2 = GeoBox(Coordinate(1.0, 1.0), Coordinate(1.0, 0.0), dt=default_test_datetime)
    assert b2 != box

    # Different time
    b2 = GeoBox(Coordinate(0.0, 1.0), Coordinate(1.0, 0.0), dt=datetime(1970, 1, 1, 1, 0))
    assert b2 != box

    assert 'test' != box


def test_geobox_hash():
    box = GeoBox(Coordinate(0.0, 1.0), Coordinate(1.0, 0.0), dt=default_test_datetime)
    b2 = GeoBox(Coordinate(0.0, 1.0), Coordinate(1.0, 0.0), dt=default_test_datetime)
    assert len({box, b2}) == 1


def test_geobox_repr():
    box = GeoBox(Coordinate(0.0, 1.0), Coordinate(1.0, 0.0))
    assert repr(box) == '<GeoBox (0.0, 1.0) - (1.0, 0.0)>'


def test_geobox_bounds():
    box = GeoBox(Coordinate(0., 1.), Coordinate(1., 0.))
    assert box.bounds == (0., 0., 1., 1.)


def test_geobox_bounding_coords():
    box = GeoBox(Coordinate(0.0, 1.0), Coordinate(1.0, 0.0))
    assert box.bounding_coords() == [
        Coordinate(0.0, 1.0), Coordinate(0.0, 0.0), Coordinate(1.0, 0.0),
        Coordinate(1.0, 1.0), Coordinate(0.0, 1.0)
    ]

    # assert self-closing
    assert box.bounding_coords()[0] == box.bounding_coords()[-1]


def test_geobox_centroid():
    box = GeoBox(Coordinate(0.0, 1.0), Coordinate(1.0, 0.0))
    assert box.centroid == Coordinate(0.5, 0.5)


def test_geobox_zero_z_preserved():
    box = GeoBox(Coordinate(0., 1., z=0.), Coordinate(1., 0., z=0.))
    assert box.centroid.z == 0.
    assert all(c.z == 0. for c in box.bounding_coords())


def test_geobox_contains():
    box = GeoBox(Coordinate(0.0, 1.0), Coordinate(1.0, 0.0))

    # Center
    assert Coordinate(0.5, 0.5) in box

    # Outside along both axes
    assert Coordinate(2.0, 0.0) not in box
    assert Coordinate(0.0, 2.0) not in box

    # on edge
    assert Coordinate(0.0, 0.5) in box


def test_geobox_contains_coordinate():
    box = GeoBox(Coordinate(0., 1.), Coordinate(1., 0.))
    assert box.contains_coordinate(Coordinate(0.5, 0.5))

    assert not box.contains_coordinate(Coordinate(2.0, 2.0))

    box = GeoBox(
        Coordinate(0., 1.), Coordinate(1., 0.),
        holes=[GeoCircle(Coordinate(0.5, 0.5), 5_000)]
    )
    assert not box.contains_coordinate(Coordinate(0.5, 0.5))


def test_geobox_copy():
    box = GeoBox(Coordinate(0., 1.), Coordinate(1., 0.))
    box_copy = box.copy()

    # Assert equality but different pointer
    assert box == box_copy
    assert box is not box_copy


def test_geobox_serialization_round_trips():
    # Boxes parse back as their polygon form
    box = GeoBox(Coordinate(0.0, 1.0), Coordinate(1.0, 0.0), dt=default_test_datetime)
    wkt_round_trip(box)
    geojson_round_trip(box)
    shapely_round_trip(box)


def test_geobox_to_wkt():
    box = GeoBox(Coordinate(0.0, 1.0), Coordinate(1.0, 0.0))
    assert box.to_wkt() == 'POLYGON((0 1,0 0,1 0,1 1,0 1))'


def test_geobox_to_shapely():
    box = GeoBox(Coordinate(0.0, 1.0), Coordinate(1.0, 0.0))
    assert box.to_shapely() == shapely.geometry.Polygon(
        [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    )


def test_geobox_from_niemeyer_geohash():
    geohash = "3fffffff"
    expected = GeoBox(
        Coordinate(-0.0054931640625, 0.0),
        Coordinate(0.0, -0.00274658203125)
    )
    assert_shape_equivalence(
        GeoBox.from_niemeyer_geohash(geohash, 16),
        expected,
        5
    )


def test_geobox_linear_rings():
    box = GeoBox(
        Coordinate(0.0, 1.0),
        Coordinate(1.0, 0.0),
    )
    rings = box.linear_rings()

    assert rings == [[
        Coordinate(0.0, 1.0),
        Coordinate(0.0, 0.0),
        Coordinate(1.0, 0.0),
        Coordinate(1.0, 1.0),
        Coordinate(0.0, 1.0),
    ]]

    # Assert self-closing
    assert rings[0][0] == rings[0][-1]


def test_geobox_to_polygon():
    box = GeoBox(Coordinate(0.0, 1.0), Coordinate(1.0, 0.0), dt=default_test_datetime)
    assert box.to_polygon() == GeoPolygon(box.bounding_coords(), dt=default_test_datetime)


def test_geobox_circumscribing_rectangle():
    box = GeoBox(Coordinate(0.0, 1.0), Coordinate(1.0, 0.0), dt=default_test_datetime)
    assert box.circumscribing_rectangle() == box


def test_geobox_circumscribing_circle():
    box = GeoBox(Coordinate(0.0, 1.0), Coordinate(1.0, 0.0), dt=default_test_datetime)
    assert box.circumscribing_circle() == GeoCircle(
        Coordinate(0.5, 0.5), 78623.19385157603, dt=default_test_datetime
    )

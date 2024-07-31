
from geostructures import *
from geostructures.typing import *


def test_geostructures_typing():
    polygon = GeoCircle(Coordinate(0., 0,), 10)
    line = GeoLineString([Coordinate(0., 0.), Coordinate(1., 1.)])
    point = GeoPoint(Coordinate(0., 0.))
    multipolygon = MultiGeoPolygon([
        GeoCircle(Coordinate(0., 0, ), 10),
        GeoCircle(Coordinate(0., 0, ), 100)
    ])
    multilinestring = MultiGeoLineString([
        GeoLineString([Coordinate(0., 0.), Coordinate(1., 1.)]),
        GeoLineString([Coordinate(1., 1.), Coordinate(2., 2.)])
    ])
    multipoint = MultiGeoPoint([
        GeoPoint(Coordinate(0., 0.)),
        GeoPoint(Coordinate(1., 1.))
    ])

    assert isinstance(polygon, GeoShape)
    assert isinstance(polygon, PolygonLike)
    assert isinstance(polygon, SingleShape)
    assert isinstance(polygon, SinglePolygon)
    assert not isinstance(polygon, PointLike)
    assert not isinstance(polygon, LineLike)
    assert not isinstance(polygon, MultiShape)

    assert isinstance(line, GeoShape)
    assert not isinstance(line, PolygonLike)
    assert isinstance(line, SingleShape)
    assert not isinstance(line, SinglePolygon)
    assert not isinstance(line, PointLike)
    assert isinstance(line, LineLike)
    assert not isinstance(line, MultiShape)

    assert isinstance(point, GeoShape)
    assert not isinstance(point, PolygonLike)
    assert isinstance(point, SingleShape)
    assert not isinstance(point, SinglePolygon)
    assert isinstance(point, PointLike)
    assert not isinstance(point, LineLike)
    assert not isinstance(point, MultiShape)

    assert isinstance(multipolygon, GeoShape)
    assert isinstance(multipolygon, PolygonLike)
    assert not isinstance(multipolygon, SingleShape)
    assert not isinstance(multipolygon, SinglePolygon)
    assert not isinstance(multipolygon, PointLike)
    assert not isinstance(multipolygon, LineLike)
    assert isinstance(multipolygon, MultiShape)

    assert isinstance(multilinestring, GeoShape)
    assert not isinstance(multilinestring, PolygonLike)
    assert not isinstance(multilinestring, SingleShape)
    assert not isinstance(multilinestring, SinglePolygon)
    assert not isinstance(multilinestring, PointLike)
    assert isinstance(multilinestring, LineLike)
    assert isinstance(multilinestring, MultiShape)

    assert isinstance(multipoint, GeoShape)
    assert not isinstance(multipoint, PolygonLike)
    assert not isinstance(multipoint, SingleShape)
    assert not isinstance(multipoint, SinglePolygon)
    assert isinstance(multipoint, PointLike)
    assert not isinstance(multipoint, LineLike)
    assert isinstance(multipoint, MultiShape)

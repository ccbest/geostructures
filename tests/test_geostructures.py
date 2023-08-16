from datetime import date, datetime, timezone
import pytest
import pytz
import shapely
from shapely import wkt

from geostructures.structures import *
from geostructures.calc import inverse_haversine_degrees
from geostructures.coordinates import Coordinate
from geostructures.time import DateInterval, TimeInterval



default_test_datetime = datetime(1970, 1, 1, 0, 0, tzinfo=timezone.utc)

d20200101 = datetime(2020, 1, 1)
d20200101_tz = pytz.timezone("America/Toronto").localize(datetime(2020, 1, 1))
d20201231 = datetime(2020, 12, 31)
d20201231_tz = pytz.timezone("America/Toronto").localize(datetime(2020, 12, 31))
r1 = TimeInterval(d20200101, d20201231)
r2 = TimeInterval(d20200101_tz, d20201231_tz)


@pytest.fixture
def geopolygon():
    return GeoPolygon(
        [
            Coordinate(0.0, 0.0), Coordinate(0.0, 1.0), Coordinate(1.0, 1.0),
            Coordinate(1.0, 0.0), Coordinate(0.0, 0.0)
        ],
        dt=default_test_datetime
    )


@pytest.fixture
def geopolygon_cycle():
    return GeoPolygon(
        [
            Coordinate(0.0, 1.0), Coordinate(1.0, 1.0), Coordinate(1.0, 0.0),
            Coordinate(0.0, 0.0), Coordinate(0.0, 1.0),
        ],
        dt=default_test_datetime
    )


@pytest.fixture
def geopolygon_reverse(geopolygon):
    return GeoPolygon(
        geopolygon.outline[::-1],
        dt=default_test_datetime
    )


@pytest.fixture
def geobox():
    return GeoBox(
        Coordinate(0.0, 1.0),
        Coordinate(1.0, 0.0),
        dt=default_test_datetime
    )


@pytest.fixture
def geocircle():
    return GeoCircle(Coordinate(0.0, 0.0), 1000, dt=default_test_datetime)


@pytest.fixture
def geoellipse():
    return GeoEllipse(Coordinate(0.0, 0.0), 1000, 500, 90, dt=default_test_datetime)


@pytest.fixture
def georing():
    return GeoRing(Coordinate(0.0, 0.0), 500, 1000, dt=default_test_datetime)


@pytest.fixture
def geowedge():
    return GeoRing(Coordinate(0.0, 0.0), 500, 1000, 90, 180, dt=default_test_datetime)


@pytest.fixture
def geolinestring():
    return GeoLineString(
        [Coordinate(0.0, 0.0), Coordinate(1.0, 0.0), Coordinate(1.0, 1.0)],
        dt=default_test_datetime
    )


@pytest.fixture
def geopoint():
    return GeoPoint(Coordinate(0.0, 0.0), dt=default_test_datetime)


def test_geoshape_start():
    geopoint = GeoPoint(Coordinate('0.0', '0.0'), dt=default_test_datetime)
    assert geopoint.start == default_test_datetime

    geopoint = GeoPoint(Coordinate('0.0', '0.0'), dt=date(1970, 1, 1))
    assert geopoint.start == date(1970, 1, 1)

    geopoint = GeoPoint(
        Coordinate('0.0', '0.0'),
        dt=TimeInterval(datetime(1970, 1, 1, 0, 0), datetime(1970, 1, 1, 1, 0))
    )
    assert geopoint.start == datetime(1970, 1, 1, 0, 0, tzinfo=timezone.utc)

    geopoint = GeoPoint(Coordinate('0.0', '0.0'), dt=DateInterval(date(1970, 1, 1), date(1970, 1, 2)))
    assert geopoint.start == date(1970, 1, 1)

    with pytest.raises(ValueError):
        _ = GeoCircle(Coordinate('0.0', '0.0'), 50).start


def test_geoshape_end():
    geopoint = GeoPoint(Coordinate('0.0', '0.0'), dt=default_test_datetime)
    assert geopoint.end == default_test_datetime

    geopoint = GeoPoint(Coordinate('0.0', '0.0'), dt=date(1970, 1, 1))
    assert geopoint.end == date(1970, 1, 1)

    geopoint = GeoPoint(
        Coordinate('0.0', '0.0'),
        dt=TimeInterval(datetime(1970, 1, 1, 0, 0), datetime(1970, 1, 1, 1, 0))
    )
    assert geopoint.end == datetime(1970, 1, 1, 1, 0, tzinfo=timezone.utc)

    geopoint = GeoPoint(Coordinate('0.0', '0.0'), dt=DateInterval(date(1970, 1, 1), date(1970, 1, 2)))
    assert geopoint.end == date(1970, 1, 2)

    with pytest.raises(ValueError):
        _ = GeoCircle(Coordinate('0.0', '0.0'), 50).end


def test_geopolygon_eq(geopolygon, geopolygon_cycle, geopolygon_reverse):
    p2 = GeoPolygon(
        [
            Coordinate(0.0, 0.0), Coordinate(0.0, 1.0), Coordinate(1.0, 1.0),
            Coordinate(1.0, 0.0), Coordinate(0.0, 0.0)
        ],
        dt=default_test_datetime
    )

    assert geopolygon == p2

    # Same coords, but rotated or reversed
    assert geopolygon == geopolygon_cycle
    assert geopolygon == geopolygon_reverse

    # Different coords - different lengths
    p2 = GeoPolygon(
        [
            Coordinate(1.0, 1.0), Coordinate(0.0, 1.0), Coordinate(1.0, 1.0),
            Coordinate(1.0, 0.0), Coordinate(0.0, 0.0)
        ],
        dt=default_test_datetime
    )
    assert geopolygon != p2


    # Different coords - same lengths
    p2 = GeoPolygon(
        [
            Coordinate(1.0, 1.0), Coordinate(0.0, 1.0),
            Coordinate(1.0, 0.0), Coordinate(0.0, 0.0)
        ],
        dt=default_test_datetime
    )
    assert geopolygon != p2

    # Different datetime
    p2 = GeoPolygon(
        [
            Coordinate(0.0, 0.0), Coordinate(0.0, 1.0), Coordinate(1.0, 1.0),
            Coordinate(1.0, 0.0), Coordinate(0.0, 0.0)
        ],
        dt=datetime(1970, 1, 1, 1, 0)
    )
    assert geopolygon != p2

    assert 'test' != geopolygon


def test_gt_to_json():
    geopoint = GeoPoint(Coordinate('0.0', '0.0'), dt=default_test_datetime)
    assert geopoint._dt_to_json() == {
        'datetime_start': default_test_datetime.isoformat(),
        'datetime_end': default_test_datetime.isoformat()
    }

    geopoint = GeoPoint(Coordinate('0.0', '0.0'), dt=date(1970, 1, 1))
    assert geopoint._dt_to_json() == {
        'date_start': date(1970, 1, 1).isoformat(),
        'date_end': date(1970, 1, 1).isoformat()
    }

    geopoint = GeoPoint(
        Coordinate('0.0', '0.0'),
        dt=TimeInterval(datetime(1970, 1, 1, 0, 0), datetime(1970, 1, 1, 1, 0))
    )
    assert geopoint._dt_to_json() == {
        'datetime_start': datetime(1970, 1, 1, 0, 0, tzinfo=timezone.utc).isoformat(),
        'datetime_end': datetime(1970, 1, 1, 1, 0, tzinfo=timezone.utc).isoformat()
    }

    geopoint = GeoPoint(Coordinate('0.0', '0.0'), dt=DateInterval(date(1970, 1, 1), date(1970, 1, 2)))
    assert geopoint._dt_to_json() == {
        'date_start': date(1970, 1, 1).isoformat(),
        'date_end': date(1970, 1, 2).isoformat()
    }

    geopoint = GeoCircle(Coordinate('0.0', '0.0'), 50)
    assert geopoint._dt_to_json() == {}


def test_geopolygon_hash(geopolygon):
    p2 = GeoPolygon(
        [
            Coordinate(0.0, 0.0), Coordinate(0.0, 1.0), Coordinate(1.0, 1.0),
            Coordinate(1.0, 0.0), Coordinate(0.0, 0.0)
        ],
        dt=default_test_datetime
    )
    assert len({geopolygon, p2}) == 1


def test_geopolygon_repr(geopolygon):
    assert repr(geopolygon) == '<GeoPolygon of 4 coordinates>'


def test_geopolygon_contains():
    # Triangle
    polygon = GeoPolygon([
        Coordinate(0.0, 0.0), Coordinate(1.0, 1.0), Coordinate(1.0, .0)
    ])
    # Way outside
    assert Coordinate(1.5, 1.5) not in polygon

    # Center along hypotenuse
    assert Coordinate(0.5, 0.5) in polygon

    # Outside, to upper left
    assert Coordinate(0.1, 0.9) not in polygon

    # 5-point Star
    polygon = GeoPolygon([
        Coordinate(0.004, 0.382), Coordinate(0.596, 0.803), Coordinate(0.364, 0.114),
        Coordinate(0.948, -0.319), Coordinate(0.221, -0.311), Coordinate(-0.01, -1),
        Coordinate(-0.228, -0.307), Coordinate(-0.954, -0.299), Coordinate(-0.362, 0.122),
        Coordinate(-0.579, 0.815), Coordinate(0.004, 0.382)
    ])
    assert Coordinate(0.0, 0.0) in polygon
    assert Coordinate(0.9, 0.1) not in polygon
    assert Coordinate(-0.9, 0.4) not in polygon
    assert Coordinate(-0.9, 0.1) not in polygon


def test_geopolygon_bounding_coords(geopolygon):
    assert geopolygon.outline == geopolygon.bounding_coords()


def test_polygon_to_geojson(geopolygon):
    assert geopolygon.to_geojson(test_prop=2) == {
        'type': 'Feature',
        'geometry': {
            'type': 'Polygon',
            'coordinates': [[x.to_float() for x in geopolygon.bounding_coords()]],
        },
        'properties': {
            'test_prop': 2,
            'datetime_start': default_test_datetime.isoformat(),
            'datetime_end': default_test_datetime.isoformat(),
        }
    }

    shapely.geometry.shape(geopolygon.to_geojson(test_prop=2)['geometry'])


def test_geopolygon_circumscribing_circle(geopolygon):
    assert geopolygon.circumscribing_circle() == GeoCircle(
        center=Coordinate(0.5, 0.5),
        radius=78626.18767687456,
        dt=default_test_datetime
    )


def test_geopolygon_circumscribing_rectangle(geopolygon):
    assert geopolygon.circumscribing_rectangle() == GeoBox(
        Coordinate(0.0, 1.0),
        Coordinate(1.0, 0.0),
        dt=default_test_datetime
    )


def test_geopolygon_to_polygon(geopolygon):
    assert geopolygon.to_polygon() == geopolygon


def test_geoshape_to_wkt(geopolygon):
    assert geopolygon.to_wkt() == 'POLYGON((0.0 0.0,0.0 1.0,1.0 1.0,1.0 0.0,0.0 0.0))'


def test_geobox_contains(geobox):
    # Center
    assert Coordinate(0.5, 0.5) in geobox

    # Outside along both axes
    assert Coordinate(2.0, 0.0) not in geobox
    assert Coordinate(0.0, 2.0) not in geobox

    # on edge
    assert Coordinate(0.0, 0.5) in geobox


def test_geobox_eq(geobox):
    b2 = GeoBox(Coordinate(0.0, 1.0), Coordinate(1.0, 0.0), dt=default_test_datetime)
    assert b2 == geobox

    # Different coords
    b2 = GeoBox(Coordinate(1.0, 1.0), Coordinate(1.0, 0.0), dt=default_test_datetime)
    assert b2 != geobox

    # Different time
    b2 = GeoBox(Coordinate(0.0, 1.0), Coordinate(1.0, 0.0), dt=datetime(1970, 1, 1, 1, 0))
    assert b2 != geobox

    assert 'test' != geobox


def test_geobox_hash(geobox):
    b2 = GeoBox(Coordinate(0.0, 1.0), Coordinate(1.0, 0.0), dt=default_test_datetime)
    assert len({geobox, b2}) == 1


def test_geobox_repr(geobox):
    assert repr(geobox) == '<GeoBox (0.0, 1.0) - (1.0, 0.0)>'


def test_geobox_bounding_coords(geobox):
    assert geobox.bounding_coords() == [
        Coordinate(0.0, 1.0), Coordinate(1.0, 1.0), Coordinate(1.0, 0.0),
        Coordinate(0.0, 0.0), Coordinate(0.0, 1.0)
    ]


def test_geobox_to_geojson(geobox):
    assert geobox.to_geojson(test_prop=2) == {
        'type': 'Feature',
        'geometry': {
            'type': 'Polygon',
            'coordinates': [[x.to_float() for x in geobox.bounding_coords()]],
        },
        'properties': {
            'test_prop': 2,
            'datetime_start': default_test_datetime.isoformat(),
            'datetime_end': default_test_datetime.isoformat(),
        }
    }

    shapely.geometry.shape(geobox.to_geojson(test_prop=2)['geometry'])


def test_geobox_to_polygon(geobox):
    assert geobox.to_polygon() == GeoPolygon(geobox.bounding_coords(), dt=default_test_datetime)


def test_geobox_to_circumscribing_rectangle(geobox):
    assert geobox.circumscribing_rectangle() == geobox


def test_geobox_to_circumscribing_circle(geobox):
    assert geobox.circumscribing_circle() == GeoCircle(Coordinate(0.5, 0.5), 78623.19385157603, dt=default_test_datetime)


def test_geobox_centroid(geobox):
    assert geobox.centroid == Coordinate(0.5, 0.5)


def test_geocircle_contains(geocircle):
    assert Coordinate(0.0, 0.0) in geocircle
    assert Coordinate(0.001, 0.001) in geocircle
    assert Coordinate(1.0, 1.0) not in geocircle


def test_geocircle_eq(geocircle):
    c2 = GeoCircle(Coordinate(0.0, 0.0), 1000, dt=default_test_datetime)
    assert c2 == geocircle

    c2 = GeoCircle(Coordinate(1.0, 1.0), 1000, dt=default_test_datetime)
    assert c2 != geocircle

    c2 = GeoCircle(Coordinate(0.0, 0.0), 2000, dt=default_test_datetime)
    assert c2 != geocircle

    c2 = GeoCircle(Coordinate(0.0, 0.0), 1000, dt=datetime(1970, 1, 1, 1, 1))
    assert c2 != geocircle

    assert 'test' != geocircle


def test_geocircle_hash(geocircle):
    c2 = GeoCircle(Coordinate(0.0, 0.0), 1000, dt=default_test_datetime)
    assert len({geocircle, c2}) == 1


def test_geocircle_repr(geocircle):
    assert repr(geocircle) == '<GeoCircle at (0.0, 0.0); radius 1000 meters>'


def test_geocircle_to_geojson(geocircle):
    assert geocircle.to_geojson(test_prop=2) == {
        'type': 'Feature',
        'geometry': {
            'type': 'Polygon',
            'coordinates': [[x.to_float() for x in geocircle.bounding_coords()]],
        },
        'properties': {
            'test_prop': 2,
            'datetime_start': default_test_datetime.isoformat(),
            'datetime_end': default_test_datetime.isoformat(),
        }
    }

    assert geocircle.to_geojson(k=10, test_prop=2) == {
        'type': 'Feature',
        'geometry': {
            'type': 'Polygon',
            'coordinates': [[x.to_float() for x in geocircle.bounding_coords(k=10)]],
        },
        'properties': {
            'test_prop': 2,
            'datetime_start': default_test_datetime.isoformat(),
            'datetime_end': default_test_datetime.isoformat(),
        }
    }

    shapely.geometry.shape(geocircle.to_geojson(k=10, test_prop=2)['geometry'])


def test_geocircle_to_polygon(geocircle):
    assert geocircle.to_polygon() == GeoPolygon(geocircle.bounding_coords(), dt=default_test_datetime)


def test_geocircle_bounding_coords(geocircle):
    assert geocircle.bounding_coords()[:5] == [
        Coordinate('0.0000000', '0.0089932'),
        Coordinate('0.0015617', '0.0088566'),
        Coordinate('0.0030759', '0.0084509'),
        Coordinate('0.0044966', '0.0077884'),
        Coordinate('0.0057807', '0.0068892')
    ]


def test_geocircle_circumscribing_rectangle(geocircle):
    assert geocircle.circumscribing_rectangle() == GeoBox(
        Coordinate(-0.0089932, 0.0089932),
        Coordinate(0.0089932, -0.0089932),
        dt=default_test_datetime
    )


def test_geocircle_circumscribing_circle(geocircle):
    assert geocircle.circumscribing_circle() == geocircle


def test_geocircle_centroid(geocircle):
    assert geocircle.centroid == geocircle.center


def test_geoellipse_contains(geoellipse):
    # Center
    assert Coordinate(0.0, 0.0) in geoellipse

    # 900 meters east
    assert Coordinate(0.0080939, 0.) in geoellipse

    # 900 meters north (outside)
    assert Coordinate(0.0, 0.0080939) not in geoellipse

    # 1000 meters east - on edge
    assert Coordinate(0.0089932, 0.) in geoellipse

    # 45-degree line
    assert Coordinate(0.005, 0.005) in GeoEllipse(Coordinate(0.0, 0.0), 1000, 1, 45)
    assert Coordinate(-0.005, -0.005) in GeoEllipse(Coordinate(0.0, 0.0), 1000, 1, 45)
    assert Coordinate(0, 0.005) not in GeoEllipse(Coordinate(0.0, 0.0), 1000, 1, 45)
    assert Coordinate(-0.005, 0.005) not in GeoEllipse(Coordinate(0.0, 0.0), 1000, 1, 45)


def test_geoellipse_eq(geoellipse):
    e2 = GeoEllipse(Coordinate(0.0, 0.0), 1000, 500, 90, dt=default_test_datetime)
    assert e2 == geoellipse

    e2 = GeoEllipse(Coordinate(1.0, 0.0), 1000, 500, 90, dt=default_test_datetime)
    assert e2 != geoellipse

    e2 = GeoEllipse(Coordinate(0.0, 0.0), 2000, 500, 90, dt=default_test_datetime)
    assert e2 != geoellipse

    e2 = GeoEllipse(Coordinate(0.0, 0.0), 1000, 600, 90, dt=default_test_datetime)
    assert e2 != geoellipse

    e2 = GeoEllipse(Coordinate(0.0, 0.0), 1000, 500, 180, dt=default_test_datetime)
    assert e2 != geoellipse

    e2 = GeoEllipse(Coordinate(0.0, 0.0), 1000, 500, 90, dt=datetime(1970, 1, 1, 1, 1))
    assert e2 != geoellipse

    assert 'test' != geoellipse


def test_geoellipse_hash(geoellipse):
    e2 = GeoEllipse(Coordinate(0.0, 0.0), 1000, 500, 90, dt=default_test_datetime)
    assert len({e2, geoellipse}) == 1


def test_geoellipse_repr(geoellipse):
    assert repr(geoellipse) == '<GeoEllipse at (0.0, 0.0); radius 1000/500; rotation 90>'


def test_geoellipse_bounding_coords(geoellipse):
    assert geoellipse.bounding_coords()[:5] == [
        Coordinate('0.0089932', '0.0000000'),
        Coordinate('0.0088586', '-0.0007750'),
        Coordinate('0.0084813', '-0.0014955'),
        Coordinate('0.0079267', '-0.0021240'),
        Coordinate('0.0072708', '-0.0026464')
    ]


def test_geoellipse_to_polygon(geoellipse):
    assert geoellipse.to_polygon() == GeoPolygon(geoellipse.bounding_coords(), dt=default_test_datetime)


def test_geoellipse_to_geojson(geoellipse):
    assert geoellipse.to_geojson(test_prop=2) == {
        'type': 'Feature',
        'geometry': {
            'type': 'Polygon',
            'coordinates': [[x.to_float() for x in geoellipse.bounding_coords()]],
        },
        'properties': {
            'test_prop': 2,
            'datetime_start': default_test_datetime.isoformat(),
            'datetime_end': default_test_datetime.isoformat(),
        }
    }

    assert geoellipse.to_geojson(k=10, test_prop=2) == {
        'type': 'Feature',
        'geometry': {
            'type': 'Polygon',
            'coordinates': [[x.to_float() for x in geoellipse.bounding_coords(k=10)]],
        },
        'properties': {
            'test_prop': 2,
            'datetime_start': default_test_datetime.isoformat(),
            'datetime_end': default_test_datetime.isoformat(),
        }
    }

    shapely.geometry.shape(geoellipse.to_geojson(k=10, test_prop=2)['geometry'])


def test_geoellipse_circumscribing_rectangle(geoellipse):
    assert geoellipse.circumscribing_rectangle() == GeoBox(
        Coordinate(-0.0089932, 0.0044966),
        Coordinate(0.0089932, -0.0044966),
        dt=default_test_datetime
    )


def test_geoellipse_circumscribing_circle(geoellipse):
    assert geoellipse.circumscribing_circle() == GeoCircle(
        geoellipse.center,
        geoellipse.major_axis,
        dt=default_test_datetime
    )


def test_geoellipse_centroid(geoellipse):
    assert geoellipse.centroid == geoellipse.center


def test_georing_contains(georing, geowedge):
    # 750 meters east
    assert inverse_haversine_degrees(georing.center, 90, 750) in georing
    assert inverse_haversine_degrees(georing.center, 90, 750) in geowedge

    # 750 meters west (outside geowedge angle)
    assert inverse_haversine_degrees(georing.center, -90, 750) in georing
    assert inverse_haversine_degrees(georing.center, -90, 750) not in geowedge

    # Centerpoint (not in shape)
    assert georing.center not in georing
    assert georing.center not in geowedge

    # Along edge (1000m east)
    assert inverse_haversine_degrees(georing.center, 90, 1000) in georing
    assert inverse_haversine_degrees(georing.center, 90, 1000) in geowedge


def test_georing_eq(geowedge):
    w2 = GeoRing(Coordinate(0.0, 0.0), 500, 1000, 90, 180, dt=default_test_datetime)
    assert w2 == geowedge

    w2 = GeoRing(Coordinate(1.0, 0.0), 500, 1000, 90, 180, dt=default_test_datetime)
    assert w2 != geowedge

    w2 = GeoRing(Coordinate(1.0, 0.0), 600, 1000, 90, 180, dt=default_test_datetime)
    assert w2 != geowedge

    w2 = GeoRing(Coordinate(1.0, 0.0), 500, 2000, 90, 180, dt=default_test_datetime)
    assert w2 != geowedge

    w2 = GeoRing(Coordinate(1.0, 0.0), 500, 1000, 80, 180, dt=default_test_datetime)
    assert w2 != geowedge

    w2 = GeoRing(Coordinate(1.0, 0.0), 500, 1000, 90, 190, dt=default_test_datetime)
    assert w2 != geowedge

    w2 = GeoRing(Coordinate(0.0, 0.0), 500, 1000, 90, 180, dt=datetime(1970, 1, 1, 1, 1))
    assert w2 != geowedge

    assert 'test' != geowedge


def test_georing_hash(geowedge):
    w2 = GeoRing(Coordinate(0.0, 0.0), 500, 1000, 90, 180, dt=default_test_datetime)
    assert len({geowedge, w2}) == 1


def test_georing_repr(geowedge, georing):
    assert repr(geowedge) == '<GeoRing at (0.0, 0.0); radii 500/1000; 90-180 degrees>'
    assert repr(georing) == '<GeoRing at (0.0, 0.0); radii 500/1000>'


def test_georing_bounding_coords(geowedge, georing):
    assert geowedge.bounding_coords()[:5] == [
        Coordinate('0.0089932', '0.0000000'),
        Coordinate('0.0088825', '-0.0014068'),
        Coordinate('0.0085531', '-0.0027791'),
        Coordinate('0.0080130', '-0.0040828'),
        Coordinate('0.0072757', '-0.0052861'),
    ]

    assert georing.bounding_coords()[:5] == [
        Coordinate('0.0000000', '0.0089932'),
        Coordinate('0.0015617', '0.0088566'),
        Coordinate('0.0030759', '0.0084509'),
        Coordinate('0.0044966', '0.0077884'),
        Coordinate('0.0057807', '0.0068892'),
    ]


def test_georing_to_geojson(georing):
    assert georing.to_geojson(test_prop=2) == {
        'type': 'Feature',
        'geometry': {
            'type': 'Polygon',
            'coordinates': [[x.to_float() for x in georing.bounding_coords()]],
        },
        'properties': {
            'test_prop': 2,
            'datetime_start': default_test_datetime.isoformat(),
            'datetime_end': default_test_datetime.isoformat(),
        }
    }

    assert georing.to_geojson(k=10, test_prop=2) == {
        'type': 'Feature',
        'geometry': {
            'type': 'Polygon',
            'coordinates': [[x.to_float() for x in georing.bounding_coords(k=10)]],
        },
        'properties': {
            'test_prop': 2,
            'datetime_start': default_test_datetime.isoformat(),
            'datetime_end': default_test_datetime.isoformat(),
        }
    }

    # Confirm shapely can read the GeoJson
    shapely.geometry.shape(georing.to_geojson(k=10, test_prop=2)['geometry'])


def test_georing_circumscribing_rectangle(georing, geowedge):

    max_lon, _ = inverse_haversine_degrees(georing.center, 90, 1000).to_float()
    min_lon, _ = inverse_haversine_degrees(georing.center, -90, 1000).to_float()
    _, max_lat = inverse_haversine_degrees(georing.center, 0, 1000).to_float()
    _, min_lat = inverse_haversine_degrees(georing.center, 180, 1000).to_float()

    assert georing.circumscribing_rectangle() == GeoBox(
        Coordinate(max_lon, max_lat),
        Coordinate(min_lon, min_lat),
        dt=default_test_datetime
    )

    assert geowedge.circumscribing_rectangle() == GeoBox(
        geowedge.center,
        Coordinate(max_lon, min_lat),
        dt=default_test_datetime
    )

def test_georing_circumscribing_circle(georing, geowedge):
    assert georing.circumscribing_circle() == GeoCircle(
        georing.centroid,
        georing.outer_radius,
        dt=default_test_datetime
    )

    assert geowedge.circumscribing_circle() == GeoCircle(
        georing.centroid,
        739.1771243016008,
        dt=default_test_datetime
    )


def test_georing_centroid(georing, geowedge):
    assert georing.centroid == georing.center

    assert geowedge.centroid == Coordinate(0.0044104, -0.0040194)


def test_georing_to_polygon(georing):
    assert georing.to_polygon() == GeoPolygon(georing.bounding_coords(), dt=default_test_datetime)


def test_geolinestring_contains(geolinestring):
    assert Coordinate(0., 0.) in geolinestring
    assert Coordinate(5., 5.) not in geolinestring


def test_geolinestring_eq(geolinestring):
    l2 = GeoLineString([Coordinate(0.0, 0.0), Coordinate(1.0, 0.0), Coordinate(1.0, 1.0)], dt=default_test_datetime)
    assert geolinestring == l2

    l2 = GeoLineString([Coordinate(0.0, 0.0), Coordinate(1.0, 0.0), Coordinate(2.0, 1.0)], dt=default_test_datetime)
    assert geolinestring != l2

    l2 = GeoLineString([Coordinate(0.0, 0.0), Coordinate(1.0, 0.0), Coordinate(1.0, 1.0)], dt=datetime(1970, 1, 1, 1, 1))
    assert geolinestring != l2

    assert 'test' != geolinestring


def test_geolinestring_hash(geolinestring):
    l2 = GeoLineString([Coordinate(0.0, 0.0), Coordinate(1.0, 0.0), Coordinate(1.0, 1.0)], dt=default_test_datetime)
    assert len({geolinestring, l2}) == 1


def test_geolinestring_repr(geolinestring):
    assert repr(geolinestring) == '<GeoLineString with 3 points>'


def test_geolinestring_bounding_coords(geolinestring):
    assert geolinestring.bounding_coords() == [*geolinestring.coords, geolinestring.coords[0]]


def test_geolinestring_to_geojson(geolinestring):

    assert geolinestring.to_geojson(test_prop=2) == {
            'type': 'Feature',
            'geometry': {
                'type': 'LineString',
                'coordinates': [x.to_float() for x in geolinestring.bounding_coords()],
            },
            'properties': {
                'test_prop': 2,
                'datetime_start': default_test_datetime.isoformat(),
                'datetime_end': default_test_datetime.isoformat(),
            }
        }

    shapely.geometry.shape(geolinestring.to_geojson(test_prop=2)['geometry'])


def test_geolinestring_circumscribing_circle(geolinestring):
    assert geolinestring.circumscribing_circle() == GeoCircle(
        Coordinate(0.6666667, 0.3333333),
        82879.43253850673,
        dt=default_test_datetime
    )


def test_geolinestring_circumscribing_rectangle(geolinestring):
    assert geolinestring.circumscribing_rectangle() == GeoBox(
        Coordinate(0.0, 1.0),
        Coordinate(1.0, 0.0),
        dt=default_test_datetime
    )


def test_geolinestring_centroid(geolinestring):
    assert geolinestring.centroid == Coordinate(0.6666667, 0.3333333)


def test_geolinestring_to_wkt(geolinestring):
    assert geolinestring.to_wkt() == 'LINESTRING(0.0 0.0,1.0 0.0,1.0 1.0)'

    wkt.loads(geolinestring.to_wkt())


def test_geolinestring_to_polygon(geolinestring):
    assert geolinestring.to_polygon() == GeoPolygon(geolinestring.bounding_coords(), dt=default_test_datetime)


def test_geopoint_contains(geopoint):
    assert geopoint not in geopoint


def test_geopoint_eq(geopoint):
    p2 = GeoPoint(Coordinate('0.0', '0.0'), dt=default_test_datetime)
    assert geopoint == p2

    p2 = GeoPoint(Coordinate('0.0', '1.0'), dt=default_test_datetime)
    assert geopoint != p2

    p2 = GeoPoint(Coordinate('0.0', '0.0'), dt=datetime(1970, 1, 1, 1, 1))
    assert geopoint != p2

    p2 = 'not a geopoint'
    assert geopoint != p2


def test_geopoint_hash(geopoint):
    p2 = GeoPoint(Coordinate('0.0', '0.0'), dt=default_test_datetime)
    assert len({geopoint, p2}) == 1

    p2 = GeoPoint(Coordinate('0.0', '1.0'), dt=default_test_datetime)
    assert len({geopoint, p2}) == 2


def test_geopoint_repr(geopoint):
    assert repr(geopoint) == "<GeoPoint at ('0.0', '0.0')>"


def test_geopoint_bounding_coords(geopoint):
    assert geopoint.bounding_coords() == [Coordinate('0.0', '0.0')]


def test_geopoint_to_geojson(geopoint):
    assert geopoint.to_geojson(test_prop=2) == {
        'type': 'Feature',
        'geometry': {
            'type': 'Point',
            'coordinates': [0.0, 0.0],
        },
        'properties': {
            'test_prop': 2,
            'datetime_start': default_test_datetime.isoformat(),
            'datetime_end': default_test_datetime.isoformat(),
        }
    }

    shapely.geometry.shape(geopoint.to_geojson(test_prop=2)['geometry'])


def test_geopoint_circumscribing_circle(geopoint):
    with pytest.raises(NotImplementedError):
        geopoint.circumscribing_circle()


def test_geopoint_circumscribing_rectangle(geopoint):
    with pytest.raises(NotImplementedError):
        geopoint.circumscribing_rectangle()


def test_geopoint_centroid(geopoint):
    assert geopoint.centroid == geopoint


def test_geopoint_to_wkt(geopoint):
    assert geopoint.to_wkt() == 'POINT((0.0 0.0))'


def test_geopoint_to_polygon(geopoint):
    with pytest.raises(NotImplementedError):
        geopoint.to_polygon()

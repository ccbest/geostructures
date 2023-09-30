from datetime import date, datetime, timezone
import pytest
import shapely
from shapely import wkt

from geostructures.structures import *
from geostructures.calc import inverse_haversine_degrees
from geostructures.coordinates import Coordinate
from geostructures.time import TimeInterval


default_test_datetime = datetime(1970, 1, 1, 0, 0, tzinfo=timezone.utc)


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


def test_geoshape_init():
    with pytest.raises(ValueError):
        _ = GeoCircle(
            Coordinate(0.0, 0.0), 1000, 
            # Hole shape itself has a hole
            holes=[GeoCircle(Coordinate(0.0, 0.0), 500, holes=[GeoCircle(Coordinate(0.0, 0.0), 250)])]
        )


def test_geoshape_start():
    geopoint = GeoPoint(Coordinate('0.0', '0.0'), dt=default_test_datetime)
    assert geopoint.start == default_test_datetime

    geopoint = GeoPoint(
        Coordinate('0.0', '0.0'),
        dt=TimeInterval(datetime(1970, 1, 1, 0, 0), datetime(1970, 1, 1, 1, 0))
    )
    assert geopoint.start == datetime(1970, 1, 1, 0, 0, tzinfo=timezone.utc)

    with pytest.raises(ValueError):
        _ = GeoCircle(Coordinate('0.0', '0.0'), 50).start


def test_geoshape_end():
    geopoint = GeoPoint(Coordinate('0.0', '0.0'), dt=default_test_datetime)
    assert geopoint.end == default_test_datetime

    geopoint = GeoPoint(
        Coordinate('0.0', '0.0'),
        dt=TimeInterval(datetime(1970, 1, 1, 0, 0), datetime(1970, 1, 1, 1, 0))
    )
    assert geopoint.end == datetime(1970, 1, 1, 1, 0, tzinfo=timezone.utc)

    with pytest.raises(ValueError):
        _ = GeoCircle(Coordinate('0.0', '0.0'), 50).end


def test_geoshape_contains_dunder():
    geopoint = GeoCircle(Coordinate('0.0', '0.0'), 500, dt=datetime(2020, 1, 1, 1))
    assert Coordinate('0.0', '0.0') in geopoint
    assert GeoPoint(Coordinate('0.0', '0.0'), dt=datetime(2020, 1, 1, 1)) in geopoint
    assert GeoPoint(Coordinate('0.0', '0.0'), dt=None) in geopoint
    assert GeoPoint(Coordinate('0.0', '0.0'), dt=datetime(2020, 1, 1, 1)) in GeoCircle(Coordinate('0.0', '0.0'), 500, dt=None)


def test_geoshape_bounding_vertices():
    poly = GeoPolygon([Coordinate(1.0, 0.0), Coordinate(1.0, 1.0), Coordinate(0.0, 0.5), Coordinate(1.0, 0.0)])
    assert poly.bounding_vertices([
        (Coordinate(1.0, 0.0), Coordinate(1.0, 1.0)),
        (Coordinate(1.0, 1.0), Coordinate(0.0, 0.5)),
        (Coordinate(0.0, 0.5), Coordinate(1.0, 0.0)),
        (Coordinate(1.0, 0.0), Coordinate(1.0, 0.0))
    ])


def test_geoshape_contains():
    # Base case
    circle_outer = GeoCircle(Coordinate(0., 0.), 5_000)
    circle_inner = GeoCircle(Coordinate(0., 0.), 2_000)
    assert circle_outer.contains(circle_inner)
    assert not circle_inner.contains(circle_outer)

    # Time bounding
    circle_outer = GeoCircle(Coordinate(0., 0.), 5_000, dt=datetime(2020, 1, 2))
    circle_inner = GeoCircle(Coordinate(0., 0.), 2_000, dt=datetime(2020, 1, 1))
    assert not circle_outer.contains(circle_inner)

    # Intersecting
    Coordinate(0.0899322, 0.0)
    circle1 = GeoCircle(Coordinate(0., 0.), 5_000)
    circle2 = GeoCircle(Coordinate(0.0899322, 0.0), 6_000)
    assert not circle1.contains(circle2)

    # inner circle full contained within hole
    circle_outer = GeoCircle(Coordinate(0., 0.,), 5_000, GeoCircle(Coordinate(0., 0.,), 4_000))
    circle_inner = GeoCircle(Coordinate(0., 0.), 2_000)
    assert not circle_outer.contains(circle_inner)


def test_geoshape_contains_time():
    geopoint = GeoPoint(Coordinate('0.0', '0.0'), dt=datetime(2020, 1, 1, 1))
    assert geopoint.contains_time(datetime(2020, 1, 1, 1))
    assert not geopoint.contains_time(datetime(2020, 1, 1, 1, 1))
    assert not geopoint.contains_time(TimeInterval(datetime(2020, 1, 1, 1), datetime(2020, 1, 1, 1, 1)))

    geopoint = GeoPoint(Coordinate('0.0', '0.0'), dt=TimeInterval(datetime(2020, 1, 1, 12), datetime(2020, 1, 3, 12)))
    assert geopoint.contains_time(datetime(2020, 1, 2))
    assert not geopoint.contains_time(datetime(2020, 1, 4, 12))
    assert geopoint.contains_time(TimeInterval(datetime(2020, 1, 1, 14),datetime(2020, 1, 1, 16)))
    assert not geopoint.contains_time(TimeInterval(datetime(2020, 1, 3, 11), datetime(2020, 1, 3, 14)))

    geopoint = GeoPoint(Coordinate('0.0', '0.0'), dt=None)
    assert not geopoint.contains_time(datetime(2020, 1, 4, 12))
    assert not geopoint.contains_time(TimeInterval(datetime(2020, 1, 1, 14), datetime(2020, 1, 1, 16)))

    with pytest.raises(ValueError):
        geopoint = GeoPoint(Coordinate('0.0', '0.0'), dt=TimeInterval(datetime(2020, 1, 1, 12), datetime(2020, 1, 3, 12)))
        geopoint.contains_time('not a date')


def test_shape_intersects():
    circle1 = GeoCircle(Coordinate(0.0, 0.0), 5_000)
    circle2 = GeoCircle(Coordinate(0.0899322, 0.0), 5_000)  # Exactly 10km to the right
    assert not circle1.intersects(circle2)
    assert not circle2.intersects(circle1)

    circle1 = GeoCircle(Coordinate(0.0, 0.0), 5_000)
    circle2 = GeoCircle(Coordinate(0.0899321, 0.0), 5_000)  # Nudged just barely to the left
    assert circle1.intersects(circle2)
    assert circle2.intersects(circle1)

    circle1 = GeoCircle(Coordinate(0.0, 0.0), 5_000)
    circle2 = GeoCircle(Coordinate(0.0, 0.0), 2_000)  # Fully contained
    assert circle1.intersects(circle2)
    assert circle2.intersects(circle1)


def test_shape_to_geojson(geocircle):
    # Assert kwargs and properties end up in the right place
    assert geocircle.to_geojson(properties={'test_prop': 1}, test_kwarg=2) == {
        'type': 'Feature',
        'geometry': {
            'type': 'Polygon',
            'coordinates': [[list(x.to_float()) for x in geocircle.bounding_coords()]],
        },
        'properties': {
            'test_prop': 1,
            'datetime_start': default_test_datetime.isoformat(),
            'datetime_end': default_test_datetime.isoformat(),
        },
        'test_kwarg': 2,
    }

    # Assert k works as intended
    assert geocircle.to_geojson(k=10) == {
        'type': 'Feature',
        'geometry': {
            'type': 'Polygon',
            'coordinates': [[list(x.to_float()) for x in geocircle.bounding_coords(k=10)]],
        },
        'properties': {
            'datetime_start': default_test_datetime.isoformat(),
            'datetime_end': default_test_datetime.isoformat(),
        }
    }


def test_geoshape_to_shapely(geobox):
    assert geobox.to_shapely() == shapely.geometry.Polygon(
        [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
    )

    outline = [
        Coordinate(0.0, 0.0), Coordinate(0.0, 1.0), Coordinate(1.0, 1.0),
        Coordinate(1.0, 0.0), Coordinate(0.0, 0.0)
    ]
    hole = GeoPolygon([
        Coordinate(0.5, 0.5), Coordinate(0.5, 0.75), Coordinate(0.75, 0.5), Coordinate(0.5, 0.5),
    ])
    polygon = GeoPolygon(outline, holes=[hole])
    expected = shapely.geometry.Polygon(
        [x.to_float() for x in outline],
        holes=[[x.to_float() for x in hole.bounding_coords()]]
    )
    assert polygon.to_shapely() == expected


def test_geoshape_set_property():
    point = GeoPoint(Coordinate('0.0', '0.0'), dt=datetime(2020, 1, 1, 12))
    assert point.properties == {
        'datetime_start': datetime(2020, 1, 1, 12, tzinfo=timezone.utc),
        'datetime_end': datetime(2020, 1, 1, 12, tzinfo=timezone.utc)
    }

    point.set_property('test_property', 1)
    assert point.properties == {
        'datetime_start': datetime(2020, 1, 1, 12, tzinfo=timezone.utc),
        'datetime_end': datetime(2020, 1, 1, 12, tzinfo=timezone.utc),
        'test_property': 1
    }


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

    # Not a geopolygon
    assert 'test' != geopolygon

    p1 = GeoPolygon(
        [
            Coordinate(0.0, 0.0), Coordinate(0.0, 1.0), Coordinate(1.0, 1.0),
            Coordinate(1.0, 0.0), Coordinate(0.0, 0.0)
        ]
    )
    p2 = GeoPolygon(
        [
            Coordinate(0.0, 0.0), Coordinate(0.0, 1.0), Coordinate(1.0, 1.0),
            Coordinate(1.0, 0.0), Coordinate(0.0, 0.0)
        ],
        holes=[GeoPolygon([
            Coordinate(0.5, 0.5), Coordinate(0.5, 0.75), Coordinate(0.75, 0.5)
        ])]
    )
    # Differing number of holes
    assert p1 != p2

    p1 = GeoPolygon(
        [
            Coordinate(0.0, 0.0), Coordinate(0.0, 1.0), Coordinate(1.0, 1.0),
            Coordinate(1.0, 0.0), Coordinate(0.0, 0.0)
        ],
        holes=[
            GeoPolygon([
                Coordinate(0.5, 0.5), Coordinate(0.5, 0.75), Coordinate(0.75, 0.5)
            ]),
            GeoPolygon([
                Coordinate(0.5, 0.5), Coordinate(0.5, 0.75), Coordinate(0.75, 0.5)
            ])
        ]
    )
    p2 = GeoPolygon(
        [
            Coordinate(0.0, 0.0), Coordinate(0.0, 1.0), Coordinate(1.0, 1.0),
            Coordinate(1.0, 0.0), Coordinate(0.0, 0.0)
        ],
        holes=[
            GeoPolygon([
                Coordinate(0.6, 0.5), Coordinate(0.5, 0.75), Coordinate(0.75, 0.5)
            ]),
            GeoPolygon([
                Coordinate(0.5, 0.5), Coordinate(0.5, 0.75), Coordinate(0.75, 0.5)
            ])
        ]
    )
    # Holes are not equal
    assert p1 != p2


def test_gt_to_json():
    geopoint = GeoPoint(Coordinate('0.0', '0.0'), dt=default_test_datetime)
    assert geopoint._dt_to_json() == {
        'datetime_start': default_test_datetime.isoformat(),
        'datetime_end': default_test_datetime.isoformat()
    }

    geopoint = GeoPoint(
        Coordinate('0.0', '0.0'),
        dt=TimeInterval(datetime(1970, 1, 1, 0, 0), datetime(1970, 1, 1, 1, 0))
    )
    assert geopoint._dt_to_json() == {
        'datetime_start': datetime(1970, 1, 1, 0, 0, tzinfo=timezone.utc).isoformat(),
        'datetime_end': datetime(1970, 1, 1, 1, 0, tzinfo=timezone.utc).isoformat()
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
        Coordinate(0.0, 0.0), Coordinate(1.0, 1.0),
        Coordinate(1.0, 0.0), Coordinate(0.0, 0.0)
    ])
    # Way outside
    assert Coordinate(1.5, 1.5) not in polygon

    # Center along hypotenuse - boundary intersection should not count
    assert Coordinate(0.5, 0.5) not in polygon

    # Nudge above to be just inside
    assert Coordinate(0.5, 0.49) in polygon

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

    # Box with hole in middle
    polygon = GeoPolygon(
        [
            Coordinate(0.0, 0.0), Coordinate(0.0, 1.0), Coordinate(1.0, 1.0),
            Coordinate(1.0, 0.0), Coordinate(0.0, 0.0)
        ],
        holes=[GeoPolygon([
            Coordinate(0.25, 0.25), Coordinate(0.25, 0.75), Coordinate(0.75, 0.75),
            Coordinate(0.75, 0.25), Coordinate(0.25, 0.25)
        ])]
    )
    assert Coordinate(0.9, 0.9) in polygon  # outside hole
    assert Coordinate(0.5, 0.5) not in polygon  # inside hole
    assert Coordinate(0.75, 0.75) in polygon  # on hole edge


def test_geopolygon_bounding_coords(geopolygon):
    assert geopolygon.outline == geopolygon.bounding_coords()

    # assert self-closing
    assert geopolygon.bounding_coords() == geopolygon.bounding_coords()


def test_geopolygon_from_geojson():
    gjson = {
        'type': 'Feature',
        'geometry': {
            'type': 'Polygon',
            'coordinates': [
                [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [0.0, 0.0]],
                [[0.25, 0.25], [0.5, 0.5], [1.0, 0.25], [0.25, 0.25]],
            ]
        },
        'properties': {'example': 'prop'}
    }
    expected = GeoPolygon(
        [Coordinate(0.0, 0.0), Coordinate(1.0, 1.0), Coordinate(2.0, 0.0), Coordinate(0.0, 0.0)],
        holes=[GeoPolygon([Coordinate(0.25, 0.25), Coordinate(0.5, 0.5), Coordinate(1.0, 0.25), Coordinate(0.25, 0.25)])],
        properties={'example': 'prop'}
    )
    assert GeoPolygon.from_geojson(gjson) == expected

    # Test custom timestamp format
    gjson = {
        'type': 'Feature',
        'geometry': {
            'type': 'Polygon',
            'coordinates': [
                [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [0.0, 0.0]],
            ]
        },
        'properties': {'datetime_start': '2020-01-01'}
    }
    expected = GeoPolygon(
        [Coordinate(0.0, 0.0), Coordinate(1.0, 1.0), Coordinate(2.0, 0.0), Coordinate(0.0, 0.0)],
        dt=datetime(2020, 1, 1)
    )
    assert GeoPolygon.from_geojson(gjson, time_format='%Y-%m-%d') == expected

    with pytest.raises(ValueError):
        bad_gjson = {
            'type': 'Feature',
            'geometry': {
                'type': 'Error',
                'coordinates': [
                    [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [0.0, 0.0]],
                ]
            },
            'properties': {'example': 'prop'}
        }
        GeoPolygon.from_geojson(bad_gjson)


def test_polygon_to_geojson(geopolygon):
    shapely.geometry.shape(geopolygon.to_geojson()['geometry'])


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


def test_geopolygon_contains_coordinate():
    polygon = GeoPolygon([Coordinate(0., 1.), Coordinate(1., 1.), Coordinate(0.5, 0.), Coordinate(0., 1.)])
    assert polygon.contains_coordinate(Coordinate(0.5, 0.5))

    # Outside bounds
    assert not polygon.contains_coordinate(Coordinate(2.0, 2.0))

    # In bounds, not in polygon
    assert not polygon.contains_coordinate(Coordinate(0.75, 0.25))

    # In hole
    polygon = GeoPolygon(
        [Coordinate(0., 1.), Coordinate(1., 1.), Coordinate(0.5, 0.), Coordinate(0., 1.)],
        holes=[GeoCircle(Coordinate(0.5, 0.5), 5_000)]
    )
    assert not polygon.contains_coordinate(Coordinate(0.5, 0.5))


def test_geopolygon_from_shapely():
    expected = GeoPolygon([Coordinate(0.0, 0.0), Coordinate(1.0, 1.0), Coordinate(2.0, 0.0), Coordinate(0.0, 0.0)])
    polygon = shapely.geometry.Polygon([(0.0, 0.0), (1.0, 1.0), (2.0, 0.0), (0.0, 0.0)])
    gpolygon = GeoPolygon.from_shapely(polygon)
    assert gpolygon == expected


def test_geopolygon_linear_rings():
    polygon = GeoPolygon(
        # Outline
        [
            Coordinate(0.0, 0.0), Coordinate(0.0, 1.0), Coordinate(1.0, 1.0),
            Coordinate(1.0, 0.0), Coordinate(0.0, 0.0)
        ],
        # Hole
        holes=[GeoPolygon([
            Coordinate(0.5, 0.5), Coordinate(0.5, 0.75), Coordinate(0.75, 0.5)
        ])]
    )
    rings = polygon.linear_rings()
    assert rings == [
        [
            Coordinate(0.0, 0.0), Coordinate(0.0, 1.0), Coordinate(1.0, 1.0),
            Coordinate(1.0, 0.0), Coordinate(0.0, 0.0)
        ],
        [
            Coordinate(0.5, 0.5), Coordinate(0.5, 0.75), Coordinate(0.75, 0.5),
            Coordinate(0.5, 0.5)
        ]
    ]

    # Assert self-closing
    assert rings[0][0] == rings[0][-1]
    assert rings[1][0] == rings[1][-1]


def test_geopolygon_to_polygon(geopolygon):
    assert geopolygon.to_polygon() == geopolygon


def test_geopolygon_from_wkt():
    wkt_str = 'POLYGON ((30.123 10, 40 40, 20 40, 10.123 20, 30.123 10))'
    assert GeoPolygon.from_wkt(wkt_str) == GeoPolygon([
        Coordinate(30.123, 10), Coordinate(40, 40), Coordinate(20, 40),
        Coordinate(10.123, 20), Coordinate(30.123, 10)
    ])

    wkt_str = 'POLYGON ((30.123 10, 40 40, 20 40, 10.123 20, 30.123 10), (15 20, 20 20, 15 15, 15 20))'
    assert GeoPolygon.from_wkt(wkt_str) == GeoPolygon(
        [
            Coordinate(30.123, 10), Coordinate(40, 40), Coordinate(20, 40),
            Coordinate(10.123, 20), Coordinate(30.123, 10)
        ],
        holes=[GeoPolygon([
            Coordinate(15, 20), Coordinate(20, 20), Coordinate(15, 15), Coordinate(15, 20)
        ])]
    )

    with pytest.raises(ValueError):
        _ = GeoPolygon.from_wkt('NOT A POLYGON')


def test_geopolygon_to_wkt():
    polygon = GeoPolygon(
        [
            Coordinate(0.0, 0.0), Coordinate(0.0, 1.0), Coordinate(1.0, 1.0),
            Coordinate(1.0, 0.0), Coordinate(0.0, 0.0)
        ],
        holes=[GeoPolygon([
            Coordinate(0.25, 0.25), Coordinate(0.25, 0.75), Coordinate(0.75, 0.75),
            Coordinate(0.75, 0.25), Coordinate(0.25, 0.25)
        ])]
    )
    assert polygon.to_wkt() == 'POLYGON((0.0 0.0,0.0 1.0,1.0 1.0,1.0 0.0,0.0 0.0),(0.25 0.25,0.25 0.75,0.75 0.75,0.75 0.25,0.25 0.25))'


def test_geoshape_to_wkt():
    box = GeoBox(Coordinate(0.0, 1.0), Coordinate(1.0, 0.0))
    assert box.to_wkt() == 'POLYGON((0.0 1.0,1.0 1.0,1.0 0.0,0.0 0.0,0.0 1.0))'


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

    # assert self-closing
    assert geobox.bounding_coords()[0] == geobox.bounding_coords()[-1]


def test_geobox_contains_coordinate(geobox):
    box = GeoBox(Coordinate(0., 1.), Coordinate(1., 0.))
    assert box.contains_coordinate(Coordinate(0.5, 0.5))

    assert not box.contains_coordinate(Coordinate(2.0, 2.0))

    box = GeoBox(
        Coordinate(0., 1.), Coordinate(1., 0.),
        holes=[GeoCircle(Coordinate(0.5, 0.5), 5_000)]
    )
    assert not box.contains_coordinate(Coordinate(0.5, 0.5))


def test_geobox_linear_rings():
    box = GeoBox(
        Coordinate(0.0, 1.0),
        Coordinate(1.0, 0.0),
    )
    rings = box.linear_rings()

    assert rings == [[
        Coordinate(0.0, 1.0),
        Coordinate(1.0, 1.0),
        Coordinate(1.0, 0.0),
        Coordinate(0.0, 0.0),
        Coordinate(0.0, 1.0),
    ]]

    # Assert self-closing
    assert rings[0][0] == rings[0][-1]


def test_geobox_to_geojson(geobox):
    shapely.geometry.shape(geobox.to_geojson()['geometry'])


def test_geobox_to_polygon(geobox):
    assert geobox.to_polygon() == GeoPolygon(geobox.bounding_coords(), dt=default_test_datetime)


def test_geobox_to_circumscribing_rectangle(geobox):
    assert geobox.circumscribing_rectangle() == geobox


def test_geobox_to_circumscribing_circle(geobox):
    assert geobox.circumscribing_circle() == GeoCircle(Coordinate(0.5, 0.5), 78623.19385157603, dt=default_test_datetime)


def test_geobox_centroid(geobox):
    assert geobox.centroid == Coordinate(0.5, 0.5)


def test_geocircle_contains_coordinate():
    circle = GeoCircle(Coordinate(0.0, 0.0), 1000)

    assert circle.contains_coordinate(Coordinate(0.0, 0.0))
    assert circle.contains_coordinate(Coordinate(0.001, 0.001))
    assert not circle.contains_coordinate(Coordinate(1.0, 1.0))

    circle = GeoCircle(Coordinate(0.0, 0.0), 1000, holes=[GeoCircle(Coordinate(0.0, 0.0), 1000)])
    assert not circle.contains_coordinate(Coordinate(0., 0.))


def test_geocircle_geojson(geocircle):
    shapely.geometry.shape(geocircle.to_geojson(k=10, test_prop=2)['geometry'])


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

    # assert self-closing
    assert geocircle.bounding_coords()[0] == geocircle.bounding_coords()[-1]


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


def test_geocircle_linear_rings(geocircle):
    rings = geocircle.linear_rings()
    assert rings == [geocircle.bounding_coords()]

    # Assert self-closing
    assert rings[0][0] == rings[0][-1]


def test_geoellipse_contains():
    ellipse = GeoEllipse(Coordinate(0.0, 0.0), 1000, 500, 90)

    # Center
    assert ellipse.contains_coordinate(Coordinate(0.0, 0.0))

    # 900 meters east
    assert ellipse.contains_coordinate(Coordinate(0.0080939, 0.))

    # 900 meters north (outside)
    assert not ellipse.contains_coordinate(Coordinate(0.0, 0.0080939))

    # 1000 meters east - on edge
    assert ellipse.contains_coordinate(Coordinate(0.0089932, 0.))

    # 45-degree line
    ellipse = GeoEllipse(Coordinate(0.0, 0.0), 1000, 1, 45)
    assert ellipse.contains_coordinate(Coordinate(0.005, 0.005))
    assert ellipse.contains_coordinate(Coordinate(-0.005, -0.005))
    assert not ellipse.contains_coordinate(Coordinate(0, 0.005))
    assert not ellipse.contains_coordinate(Coordinate(-0.005, 0.005))

    # Hole
    ellipse = GeoEllipse(Coordinate(0.0, 0.0), 1000, 500, 90, holes=[GeoCircle(Coordinate(0., 0.), 200)])
    assert not ellipse.contains_coordinate(Coordinate(0., 0))

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

    # assert self-closing
    assert geoellipse.bounding_coords()[0] == geoellipse.bounding_coords()[-1]


def test_geoellipse_linear_rings(geoellipse):
    rings = geoellipse.linear_rings()
    assert rings == [geoellipse.bounding_coords()]

    # Assert self-closing
    assert rings[0][0] == rings[0][-1]


def test_geoellipse_to_polygon(geoellipse):
    assert geoellipse.to_polygon() == GeoPolygon(geoellipse.bounding_coords(), dt=default_test_datetime)


def test_geoellipse_to_geojson(geoellipse):
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
    ring = GeoRing(Coordinate(0.0, 0.0), 500, 1000)
    wedge = GeoRing(Coordinate(0.0, 0.0), 500, 1000, 90, 180)
    # 750 meters east
    assert ring.contains_coordinate(Coordinate(0.0067449, 0.0))
    assert wedge.contains_coordinate(Coordinate(0.0067449, 0.0))

    # 750 meters west (outside geowedge angle)
    assert ring.contains_coordinate(Coordinate(-0.0067449, 0.0))
    assert not wedge.contains_coordinate(Coordinate(-0.0067449, 0.0))

    # Centerpoint (not in shape)
    assert not ring.contains_coordinate(Coordinate(0.0, 0.0))
    assert not wedge.contains_coordinate(Coordinate(0.0, 0.0))

    # Along edge (1000m east)
    assert ring.contains_coordinate(Coordinate(0.0089932, 0.0))
    assert wedge.contains_coordinate(Coordinate(0.0089932, 0.0))

    ring = GeoRing(Coordinate(0.0, 0.0), 500, 1000, holes=[GeoCircle(Coordinate(0.0067449, 0.0), 200)])
    assert not ring.contains_coordinate(Coordinate(0.0067449, 0.0))


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

    # Assert self-closing
    assert geowedge.bounding_coords()[0] == geowedge.bounding_coords()[-1]

    assert georing.bounding_coords()[:5] == [
        Coordinate('0.0000000', '0.0089932'),
        Coordinate('0.0015617', '0.0088566'),
        Coordinate('0.0030759', '0.0084509'),
        Coordinate('0.0044966', '0.0077884'),
        Coordinate('0.0057807', '0.0068892'),
    ]

    # assert self-closing
    assert georing.bounding_coords()[0] == georing.bounding_coords()[-1]


def test_georing_to_geojson(georing):
    shapely.geometry.shape(georing.to_geojson()['geometry'])


def test_georing_circumscribing_rectangle(georing, geowedge):

    max_lon, _ = inverse_haversine_degrees(georing.center, 90, 1000).to_float()
    min_lon, _ = inverse_haversine_degrees(georing.center, -90, 1000).to_float()
    _, max_lat = inverse_haversine_degrees(georing.center, 0, 1000).to_float()
    _, min_lat = inverse_haversine_degrees(georing.center, 180, 1000).to_float()

    assert georing.circumscribing_rectangle() == GeoBox(
        Coordinate(min_lon, max_lat),
        Coordinate(max_lon, min_lat),
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
        Coordinate(0.0044104, -0.0040194),
        739.1771243016008,
        dt=default_test_datetime
    )


def test_georing_centroid(georing, geowedge):
    assert georing.centroid == georing.center

    assert geowedge.centroid == Coordinate(0.0044104, -0.0040194)


def test_georing_linear_rings(georing, geowedge):
    rings = georing.linear_rings()
    assert len(rings) == 2  # should have outer and inner shell
    assert rings[0][:5] == [
        Coordinate(0.0, 0.0089932),
        Coordinate(0.0015617, 0.0088566),
        Coordinate(0.0030759, 0.0084509),
        Coordinate(0.0044966, 0.0077884),
        Coordinate(0.0057807, 0.0068892)
    ]
    # Assert self-closing
    assert rings[0][0] == rings[0][-1]
    assert rings[1][0] == rings[1][-1]

    rings = geowedge.linear_rings()
    assert len(rings) == 1
    assert rings[0][0] == rings[0][-1]


def test_georing_to_wkt(georing, geowedge):
    assert georing.to_wkt() == 'POLYGON((0.0 0.0089932,0.0015617 0.0088566,0.0030759 0.0084509,0.0044966 0.0077884,0.0057807 0.0068892,0.0068892 0.0057807,0.0077884 0.0044966,0.0084509 0.0030759,0.0088566 0.0015617,0.0089932 0.0,0.0088566 -0.0015617,0.0084509 -0.0030759,0.0077884 -0.0044966,0.0068892 -0.0057807,0.0057807 -0.0068892,0.0044966 -0.0077884,0.0030759 -0.0084509,0.0015617 -0.0088566,0.0 -0.0089932,-0.0015617 -0.0088566,-0.0030759 -0.0084509,-0.0044966 -0.0077884,-0.0057807 -0.0068892,-0.0068892 -0.0057807,-0.0077884 -0.0044966,-0.0084509 -0.0030759,-0.0088566 -0.0015617,-0.0089932 -0.0,-0.0088566 0.0015617,-0.0084509 0.0030759,-0.0077884 0.0044966,-0.0068892 0.0057807,-0.0057807 0.0068892,-0.0044966 0.0077884,-0.0030759 0.0084509,-0.0015617 0.0088566,0.0 0.0089932), (0.0 0.0044966,0.0007808 0.0044283,0.0015379 0.0042254,0.0022483 0.0038942,0.0028904 0.0034446,0.0034446 0.0028904,0.0038942 0.0022483,0.0042254 0.0015379,0.0044283 0.0007808,0.0044966 0.0,0.0044283 -0.0007808,0.0042254 -0.0015379,0.0038942 -0.0022483,0.0034446 -0.0028904,0.0028904 -0.0034446,0.0022483 -0.0038942,0.0015379 -0.0042254,0.0007808 -0.0044283,0.0 -0.0044966,-0.0007808 -0.0044283,-0.0015379 -0.0042254,-0.0022483 -0.0038942,-0.0028904 -0.0034446,-0.0034446 -0.0028904,-0.0038942 -0.0022483,-0.0042254 -0.0015379,-0.0044283 -0.0007808,-0.0044966 -0.0,-0.0044283 0.0007808,-0.0042254 0.0015379,-0.0038942 0.0022483,-0.0034446 0.0028904,-0.0028904 0.0034446,-0.0022483 0.0038942,-0.0015379 0.0042254,-0.0007808 0.0044283,0.0 0.0044966))'
    assert geowedge.to_wkt() == 'POLYGON((0.0089932 0.0,0.0088825 -0.0014068,0.0085531 -0.0027791,0.008013 -0.0040828,0.0072757 -0.0052861,0.0063592 -0.0063592,0.0052861 -0.0072757,0.0040828 -0.008013,0.0027791 -0.0085531,0.0014068 -0.0088825,0.0 -0.0089932,0.0 -0.0044966,0.0007034 -0.0044412,0.0013895 -0.0042765,0.0020414 -0.0040065,0.002643 -0.0036378,0.0031796 -0.0031796,0.0036378 -0.002643,0.0040065 -0.0020414,0.0042765 -0.0013895,0.0044412 -0.0007034,0.0044966 0.0,0.0089932 0.0))'


def test_georing_to_polygon(georing):
    assert georing.to_polygon() == GeoPolygon(georing.bounding_coords(), dt=default_test_datetime)


def test_geolinestring_contains_dunder(geolinestring):
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
    assert geolinestring.bounding_coords() == [
        Coordinate(0.0, 0.0), Coordinate(1.0, 0.0), Coordinate(1.0, 1.0)
    ]


def test_geolinestring_contains():
    ls = GeoLineString([Coordinate(0., 0.), Coordinate(1., 1.), Coordinate(2., 2.)])
    shape = GeoCircle(Coordinate(0., 0.), 500)
    assert not ls.contains(shape)

    ls2 = GeoLineString([Coordinate(0., 0.), Coordinate(1., 1.)])
    assert ls.contains(ls2)

    ls3 = GeoLineString([Coordinate(0., 1.), Coordinate(0., 0.), Coordinate(1.,1.)])
    assert not ls.contains(ls3)

    point = GeoPoint(Coordinate(0., 0.))
    assert ls.contains(point)

    point2 = GeoPoint(Coordinate(1., 0.))
    assert not ls.contains(point2)


def test_geolinestring_from_geojson():
    gls = {
        'type': 'Feature',
        'geometry': {
            'type': 'LineString',
            'coordinates': [[0.0, 0.0], [1.0, 1.5], [2.0, 2.0]]
        },
        'properties': {'example': 'prop'}
    }
    expected = GeoLineString(
        [Coordinate(0.0, 0.0), Coordinate(1.0, 1.5), Coordinate(2.0, 2.0)],
        properties={'example': 'prop'}
    )
    assert GeoLineString.from_geojson(gls) == expected

    with pytest.raises(ValueError):
        bad_gjson = {
            'type': 'Feature',
            'geometry': {
                'type': 'Error',
                'coordinates': [
                    [0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [0.0, 0.0],
                ]
            },
            'properties': {'example': 'prop'}
        }
        GeoLineString.from_geojson(bad_gjson)


def test_geolinestring_intersects():
    ls = GeoLineString([Coordinate(0.0, 0.0), Coordinate(1.0, 1.0)])
    circle = GeoCircle(Coordinate(0.0, 0.0), 5_000)
    assert ls.intersects(circle)

    ls = GeoLineString([Coordinate(0.0, 0.0), Coordinate(1.0, 1.0)])
    circle = GeoCircle(Coordinate(0.0, 0.0), 5_000)
    assert ls.intersects(circle)


def test_geolinestring_linear_rings(geolinestring):
    with pytest.raises(NotImplementedError):
        _ = geolinestring.linear_rings()


def test_geolinestring_to_geojson(geolinestring):

    assert geolinestring.to_geojson(properties={'test_prop': 2}, test_kwarg=1) == {
        'type': 'Feature',
        'geometry': {
            'type': 'LineString',
            'coordinates': [list(x.to_float()) for x in geolinestring.bounding_coords()],
        },
        'properties': {
            'test_prop': 2,
            'datetime_start': default_test_datetime.isoformat(),
            'datetime_end': default_test_datetime.isoformat(),
        },
        'test_kwarg': 1
    }


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


def test_geolinestring_from_shapely():
    expected = GeoLineString([Coordinate(0.0, 0.0), Coordinate(1.0, 1.0), Coordinate(2.0, 2.0)])
    ls = shapely.geometry.LineString([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)])
    gls = GeoLineString.from_shapely(ls)
    assert gls == expected


def test_geolinestring_from_wkt():
    wkt_str = 'LINESTRING (30.123 10, 10 30.123, 40 40)'
    assert GeoLineString.from_wkt(wkt_str) == GeoLineString([
        Coordinate(30.123, 10), Coordinate(10, 30.123), Coordinate(40, 40)
    ])

    wkt_str = 'LINESTRING(30 10,10 30,40 40)'
    assert GeoLineString.from_wkt(wkt_str) == GeoLineString([
        Coordinate(30, 10), Coordinate(10, 30), Coordinate(40, 40)
    ])

    with pytest.raises(ValueError):
        _ = GeoLineString.from_wkt('BAD WKT')

    with pytest.raises(ValueError):
        _ = GeoLineString.from_wkt('LINESTRING(30 10,10 30,40 40) (1 2, 3 4)')


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
    assert repr(geopoint) == "<GeoPoint at (0.0, 0.0)>"


def test_geopoint_bounding_coords(geopoint):
    with pytest.raises(NotImplementedError):
        _ = geopoint.bounding_coords()


def test_geopoint_from_geojson():
    gpoint = {
        'type': 'Feature',
        'geometry': {
            'type': 'Point',
            'coordinates': [1.0, 0.0]
        },
        'properties': {'example': 'prop'}
    }
    expected = GeoPoint(
        Coordinate(1.0, 0.0),
        properties={'example': 'prop'}
    )
    assert GeoPoint.from_geojson(gpoint) == expected

    with pytest.raises(ValueError):
        bad_gjson = {
            'type': 'Feature',
            'geometry': {
                'type': 'Error',
                'coordinates': [0.0, 0.0],
            },
            'properties': {'example': 'prop'}
        }
        GeoPoint.from_geojson(bad_gjson)


def test_geopoint_from_shapely():
    expected = GeoPoint(Coordinate(0.0, 0.0))
    point = shapely.geometry.Point(0.0, 0.0)
    gpoint = GeoPoint.from_shapely(point)
    assert gpoint == expected


def test_geopoint_to_geojson(geopoint):
    assert geopoint.to_geojson(properties={'test_prop': 2}, test_kwarg=1) == {
        'type': 'Feature',
        'geometry': {
            'type': 'Point',
            'coordinates': [0.0, 0.0],
        },
        'properties': {
            'test_prop': 2,
            'datetime_start': default_test_datetime.isoformat(),
            'datetime_end': default_test_datetime.isoformat(),
        },
        'test_kwarg': 1
    }

    shapely.geometry.shape(geopoint.to_geojson(test_prop=2)['geometry'])


def test_geopoint_to_shapely(geopoint):
    assert geopoint.to_shapely() == shapely.Point(0.0, 0.0)


def test_geopoint_circumscribing_circle(geopoint):
    with pytest.raises(NotImplementedError):
        geopoint.circumscribing_circle()


def test_geopoint_circumscribing_rectangle(geopoint):
    with pytest.raises(NotImplementedError):
        geopoint.circumscribing_rectangle()


def test_geopoint_centroid(geopoint):
    assert geopoint.centroid == geopoint.center


def test_geopoint_linear_rings(geopoint):
    with pytest.raises(NotImplementedError):
        _ = geopoint.linear_rings()


def test_geopoint_from_wkt():
    wkt_str = 'POINT (1.0 1.0)'
    assert GeoPoint.from_wkt(wkt_str) == GeoPoint(Coordinate(1.0, 1.0))

    wkt_str = 'POINT(1.0 1.0)'
    assert GeoPoint.from_wkt(wkt_str) == GeoPoint(Coordinate(1.0, 1.0))

    with pytest.raises(ValueError):
        _ = GeoPoint.from_wkt('NOT WKT')


def test_geopoint_to_wkt(geopoint):
    assert geopoint.to_wkt() == 'POINT(0.0 0.0)'


def test_geopoint_to_polygon(geopoint):
    with pytest.raises(NotImplementedError):
        geopoint.to_polygon()

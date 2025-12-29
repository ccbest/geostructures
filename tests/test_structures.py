from datetime import datetime, timezone

import numpy as np
import pytest
from pytest import approx
import shapely
from shapely import wkt

from build.lib.geostructures.distance import destination_point
from geostructures.structures import *
from geostructures.coordinates import Coordinate
from geostructures.multistructures import *
from geostructures.utils.functions import round_half_up
from geostructures.time import TimeInterval

from tests import assert_shape_equivalence
from tests.functions import assert_coordinates_equal, assert_geopolygons_equal


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


def test_geoshape_area():
    assert GeoBox(Coordinate(0.0, 1.0), Coordinate(1., 0.)).area == 12308778361.469452


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


def test_geoshape_has_m():
    geo = GeoCircle(Coordinate('0.0', '0.0'), 500, dt=default_test_datetime)
    assert not geo.has_m

    geo = GeoPolygon([
        Coordinate('0.0', '0.0', m=1),
        Coordinate('1.0', '1.0', m=1),
        Coordinate('1.0', '0.0', m=1),
        Coordinate('0.0', '0.0', m=1),
    ])
    assert geo.has_m


def test_geoshape_has_z():
    geo = GeoCircle(Coordinate('0.0', '0.0'), 500)
    assert not geo.has_z

    geo = GeoCircle(Coordinate('0.0', '0.0', z=1), 500)
    assert geo.has_z


def test_geoshape_volume():
    assert GeoBox(
        Coordinate(0.0, 1.0),
        Coordinate(1., 0.),
        dt=TimeInterval(datetime(2020, 1, 1, 1, 1, 1), datetime(2020, 1, 1, 1, 1, 2))
    ).volume == 12308778361.469452

    assert GeoBox(
        Coordinate(0.0, 1.0),
        Coordinate(1., 0.),
        dt=TimeInterval(datetime(2020, 1, 1, 1, 1, 1), datetime(2020, 1, 1, 1, 1, 3))
    ).volume == 24617556722.938904

    assert GeoBox(Coordinate(0.0, 1.0), Coordinate(1., 0.)).volume == 0.
    assert GeoBox(Coordinate(0.0, 1.0), Coordinate(1., 0.), dt=datetime(2020, 1, 1)).volume == 0.


def test_geoshape_contains_dunder():
    geopoint = GeoCircle(Coordinate('0.0', '0.0'), 500, dt=datetime(2020, 1, 1, 1))
    assert Coordinate('0.0', '0.0') in geopoint
    assert GeoPoint(Coordinate('0.0', '0.0'), dt=datetime(2020, 1, 1, 1)) in geopoint
    assert GeoPoint(Coordinate('0.0', '0.0'), dt=None) in geopoint
    assert GeoPoint(Coordinate('0.0', '0.0'), dt=datetime(2020, 1, 1, 1)) in GeoCircle(Coordinate('0.0', '0.0'), 500, dt=None)


def test_geoshape_bounding_edges():
    poly = GeoPolygon([Coordinate(1.0, 0.0), Coordinate(1.0, 1.0), Coordinate(0.0, 0.5), Coordinate(1.0, 0.0)])
    assert poly.bounding_edges() == [
        (Coordinate(1.0, 0.0), Coordinate(1.0, 1.0)),
        (Coordinate(1.0, 1.0), Coordinate(0.0, 0.5)),
        (Coordinate(0.0, 0.5), Coordinate(1.0, 0.0)),
        (Coordinate(1.0, 0.0), Coordinate(1.0, 0.0))
    ]


def test_geoshape_intersects():
    # Base case
    circle1 = GeoCircle(Coordinate(0.0, 0.0), 5_000)
    circle2 = GeoCircle(Coordinate(0.0, 0.0), 5_000)
    assert circle1.intersects(circle2)

    # Intersecting datetimes
    circle1 = GeoCircle(Coordinate(0.0, 0.0), 5_000, dt=datetime(2020, 1, 1))
    circle2 = GeoCircle(Coordinate(0.0, 0.0), 5_000, dt=datetime(2020, 1, 1))
    assert circle1.intersects(circle2)

    # Non-intersecting datetimes
    circle1 = GeoCircle(Coordinate(0.0, 0.0), 5_000, dt=datetime(2020, 1, 2))
    circle2 = GeoCircle(Coordinate(0.0, 0.0), 5_000, dt=datetime(2020, 1, 1))
    assert not circle1.intersects(circle2)



def test_baseshape_contains_coordinate():

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


def test_baseshape_contains_shape():
    # Base case
    circle_outer = GeoCircle(Coordinate(0., 0.), 5_000)
    circle_inner = GeoCircle(Coordinate(0., 0.), 2_000)
    assert circle_outer.contains_shape(circle_inner)
    assert not circle_inner.contains_shape(circle_outer)

    # Intersecting, not containing
    circle1 = GeoCircle(Coordinate(0., 0.), 5_000)
    circle2 = GeoCircle(Coordinate(0.0899322, 0.0), 6_000)
    assert not circle1.contains_shape(circle2)

    # inner circle fully contained within hole
    circle_outer = GeoCircle(Coordinate(0., 0.,), 5_000, holes=[GeoCircle(Coordinate(0., 0.,), 4_000)])
    circle_inner = GeoCircle(Coordinate(0., 0.), 2_000)
    assert not circle_outer.contains_shape(circle_inner)

    # Verify it works for multishapes
    mp = MultiGeoPoint([GeoPoint(Coordinate(0., 0.)), GeoPoint(Coordinate(0.00001, 0.00001))])
    assert circle1.contains(mp)

    mp = MultiGeoPoint([GeoPoint(Coordinate(0., 0.)), GeoPoint(Coordinate(1.0, 1.0))])
    assert not circle1.contains(mp)


def test_shape_to_geojson():
    geocircle = GeoCircle(Coordinate(0.0, 0.0), 1000, dt=default_test_datetime)
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

    assert geocircle.to_geojson()['geometry'] == geocircle.__geo_interface__


def test_geoshape_intersects_shape():
    shape = GeoCircle(Coordinate(0., 0.), 5_000)
    assert shape.intersects_shape(GeoPoint(Coordinate(0., 0.)))
    assert shape.intersects_shape(
        MultiGeoPoint([
            GeoPoint(Coordinate(0., 0.)),
            GeoPoint(Coordinate(0.001, 0.001))
        ])
    )
    assert not shape.intersects_shape(
        MultiGeoPoint([
            GeoPoint(Coordinate(1., 0.)),
            GeoPoint(Coordinate(1., 0.001))
        ])
    )


def test_geoshape_to_shapely(geobox):
    assert geobox.to_shapely() == shapely.geometry.Polygon(
        [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    )

    outline = [
        Coordinate(0.0, 0.0), Coordinate(1.0, 0.0), Coordinate(1.0, 1.0),
        Coordinate(0.0, 1.0), Coordinate(0.0, 0.0)
    ]
    hole = GeoPolygon([
        Coordinate(0.5, 0.5), Coordinate(0.75, 0.5), Coordinate(0.5, 0.75), Coordinate(0.5, 0.5),
    ])
    polygon = GeoPolygon(outline, holes=[hole])
    expected = shapely.geometry.Polygon(
        [x.to_float() for x in outline],
        holes=[list(reversed([x.to_float() for x in hole.bounding_coords()]))]
    )
    assert polygon.to_shapely() == expected




def test_geopolygon_eq(geopolygon, geopolygon_cycle, geopolygon_reverse):
    p2 = GeoPolygon(
        [
            Coordinate(0.0, 0.0), Coordinate(0.0, 1.0), Coordinate(1.0, 1.0),
            Coordinate(1.0, 0.0), Coordinate(0.0, 0.0)
        ],
        dt=default_test_datetime
    )

    assert geopolygon == p2

    # Same vertices, but rotated or reversed
    assert geopolygon == geopolygon_cycle
    assert geopolygon == geopolygon_reverse

    # Different vertices - different lengths
    p2 = GeoPolygon(
        [
            Coordinate(1.0, 1.0), Coordinate(0.0, 1.0), Coordinate(1.0, 1.0),
            Coordinate(1.0, 0.0), Coordinate(0.0, 0.0)
        ],
        dt=default_test_datetime
    )
    assert geopolygon != p2


    # Different vertices - same lengths
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


def test_geopolygon_bounds():
    poly = GeoPolygon(
        [
            Coordinate(0.0, 0.0), Coordinate(0.0, 1.0), Coordinate(1.0, 1.0),
            Coordinate(1.0, 0.0), Coordinate(0.0, 0.0)
        ],
    )
    assert poly.bounds == (0., 0., 1., 1.)


def test_geopolygon_bounding_coords(geopolygon):
    assert geopolygon.outline == geopolygon.bounding_coords()

    # assert self-closing
    assert geopolygon.bounding_coords() == geopolygon.bounding_coords()


def test_geopolygon_copy():
    poly = GeoPolygon(
        [Coordinate(0.0, 0.0), Coordinate(1.0, 1.0), Coordinate(2.0, 0.0), Coordinate(0.0, 0.0)],
        holes=[GeoPolygon([Coordinate(0.25, 0.25), Coordinate(0.5, 0.5), Coordinate(1.0, 0.25), Coordinate(0.25, 0.25)])],
        properties={'example': 'prop'}
    )
    poly_copy = poly.copy()

    # Assert equality but different pointer
    assert poly == poly_copy
    assert poly is not poly_copy


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

    # Only geo interface
    gjson = gjson['geometry']
    expected = GeoPolygon(
        [Coordinate(0.0, 0.0), Coordinate(1.0, 1.0), Coordinate(2.0, 0.0), Coordinate(0.0, 0.0)],
        holes=[GeoPolygon([Coordinate(0.25, 0.25), Coordinate(0.5, 0.5), Coordinate(1.0, 0.25), Coordinate(0.25, 0.25)])],
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


def test_geopolygon_from_h3_geohash():
    geohash = "88754e6499fffff"
    expected = GeoPolygon(
        [
            Coordinate(-0.000695, -0.005249),
            Coordinate(0.001237, -0.001207),
            Coordinate(-0.000777, 0.002231),
            Coordinate(-0.004725, 0.001630),
            Coordinate(-0.006658, -0.002410),
            Coordinate(-0.004643, -0.005850),
            Coordinate(-0.000695, -0.005249)
        ]
    )
    assert_shape_equivalence(
        GeoPolygon.from_h3_geohash(geohash),
        expected,
        5
    )


def test_geopolygon_from_niemeyer_geohash():
    geohash = "3fffffff"
    expected = GeoBox(
        Coordinate(-0.005493, 0.0),
        Coordinate(0.0, -0.005493)
    ).to_polygon()
    assert_shape_equivalence(
        GeoPolygon.from_niemeyer_geohash(geohash, 16),
        expected,
        5
    )


def test_polygon_to_geojson(geopolygon):
    shapely.geometry.shape(geopolygon.to_geojson()['geometry'])


def test_geopolygon_circumscribing_circle(geopolygon):
    gc = geopolygon.circumscribing_circle()
    assert round_half_up(gc.center.latitude, 4) == 0.5
    assert round_half_up(gc.center.longitude, 4) == 0.5
    assert round_half_up(gc.radius, 0) == 78625


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
            Coordinate(0.0, 0.0), Coordinate(1.0, 0.0), Coordinate(1.0, 1.0),
            Coordinate(0.0, 1.0), Coordinate(0.0, 0.0)
        ],
        # Hole
        holes=[
            GeoPolygon([Coordinate(0.5, 0.5), Coordinate(0.75, 0.5), Coordinate(0.5, 0.75), Coordinate(0.5, 0.5)])
        ]
    )
    rings = polygon.linear_rings()
    assert rings == [
        [
            Coordinate(0.0, 0.0), Coordinate(1.0, 0.0), Coordinate(1.0, 1.0),
            Coordinate(0.0, 1.0), Coordinate(0.0, 0.0)
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

    wkt_str = 'POLYGON ZM ((30.123 10 1 2, 40 40 1 2, 20 40 1 2, 10.123 20 1 2, 30.123 10 1 2))'
    poly = GeoPolygon.from_wkt(wkt_str)
    assert [x.z for x in poly.outline] == [1., 1., 1., 1., 1.]
    assert [x.m for x in poly.outline] == [2., 2., 2., 2., 2.]

    # Test varying the ZM order
    wkt_str = 'POLYGON MZ ((30.123 10 1 2, 40 40 1 2, 20 40 1 2, 10.123 20 1 2, 30.123 10 1 2))'
    poly = GeoPolygon.from_wkt(wkt_str)
    assert [x.m for x in poly.outline] == [1., 1., 1., 1., 1.]
    assert [x.z for x in poly.outline] == [2., 2., 2., 2., 2.]

    # Test missing ZM
    wkt_str = 'POLYGON ((30.123 10 1 2, 40 40 1 2, 20 40 1 2, 10.123 20 1 2, 30.123 10 1 2))'
    poly = GeoPolygon.from_wkt(wkt_str)
    assert [x.z for x in poly.outline] == [1., 1., 1., 1., 1.]
    assert [x.m for x in poly.outline] == [2., 2., 2., 2., 2.]

    # Test only Z
    wkt_str = 'POLYGON Z ((30.123 10 1, 40 40 1, 20 40 1, 10.123 20 1, 30.123 10 1))'
    poly = GeoPolygon.from_wkt(wkt_str)
    assert [x.z for x in poly.outline] == [1., 1., 1., 1., 1.]
    assert [x.m for x in poly.outline] == [None, None, None, None, None]

    # Test only M
    wkt_str = 'POLYGON M ((30.123 10 1, 40 40 1, 20 40 1, 10.123 20 1, 30.123 10 1))'
    poly = GeoPolygon.from_wkt(wkt_str)
    assert [x.m for x in poly.outline] == [1., 1., 1., 1., 1.]
    assert [x.z for x in poly.outline] == [None, None, None, None, None]

    with pytest.raises(ValueError):
        _ = GeoPolygon.from_wkt('NOT A POLYGON')


def test_geoshape_to_wkt():
    box = GeoBox(Coordinate(0.0, 1.0), Coordinate(1.0, 0.0))
    assert box.to_wkt() == 'POLYGON((0 1,0 0,1 0,1 1,0 1))'

    polygon = GeoPolygon(
        [
            Coordinate(0.0, 0.0), Coordinate(1.0, 0.0), Coordinate(1.0, 1.0),
            Coordinate(0.0, 1.0), Coordinate(0.0, 0.0)
        ],
        holes=[GeoPolygon([
            Coordinate(0.25, 0.25), Coordinate(0.75, 0.25), Coordinate(0.75, 0.75),
            Coordinate(0.25, 0.75), Coordinate(0.25, 0.25)
        ])]
    )
    assert polygon.to_wkt() == 'POLYGON((0 0,1 0,1 1,0 1,0 0), (0.25 0.25,0.25 0.75,0.75 0.75,0.75 0.25,0.25 0.25))'



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

    # Different vertices
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


def test_geobox_bounds():
    box = GeoBox(Coordinate(0., 1.), Coordinate(1., 0.))
    assert box.bounds == (0., 0., 1., 1.)


def test_geobox_bounding_coords(geobox):
    assert geobox.bounding_coords() == [
        Coordinate(0.0, 1.0), Coordinate(0.0, 0.0), Coordinate(1.0, 0.0),
        Coordinate(1.0, 1.0), Coordinate(0.0, 1.0)
    ]

    # assert self-closing
    assert geobox.bounding_coords()[0] == geobox.bounding_coords()[-1]


def test_geobox_copy():
    box = GeoBox(Coordinate(0., 1.), Coordinate(1., 0.))
    box_copy = box.copy()

    # Assert equality but different pointer
    assert box == box_copy
    assert box is not box_copy


def test_geobox_contains_coordinate(geobox):
    box = GeoBox(Coordinate(0., 1.), Coordinate(1., 0.))
    assert box.contains_coordinate(Coordinate(0.5, 0.5))

    assert not box.contains_coordinate(Coordinate(2.0, 2.0))

    box = GeoBox(
        Coordinate(0., 1.), Coordinate(1., 0.),
        holes=[GeoCircle(Coordinate(0.5, 0.5), 5_000)]
    )
    assert not box.contains_coordinate(Coordinate(0.5, 0.5))


def test_geobox_from_niemeyer_geohash():
    geohash = "3fffffff"
    expected = GeoBox(
        Coordinate(-0.005493, 0.0),
        Coordinate(0.0, -0.005493)
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


def test_geocircle_geojson():
    geocircle = GeoCircle(Coordinate(0.0, 0.0), 1000, dt=default_test_datetime)

    shapely.geometry.shape(geocircle.to_geojson(k=10, test_prop=2)['geometry'])


def test_geocircle_eq():
    geocircle = GeoCircle(Coordinate(0.0, 0.0), 1000, dt=default_test_datetime)

    c2 = GeoCircle(Coordinate(0.0, 0.0), 1000, dt=default_test_datetime)
    assert c2 == geocircle

    c2 = GeoCircle(Coordinate(1.0, 1.0), 1000, dt=default_test_datetime)
    assert c2 != geocircle

    c2 = GeoCircle(Coordinate(0.0, 0.0), 2000, dt=default_test_datetime)
    assert c2 != geocircle

    c2 = GeoCircle(Coordinate(0.0, 0.0), 1000, dt=datetime(1970, 1, 1, 1, 1))
    assert c2 != geocircle

    assert 'test' != geocircle


def test_geocircle_hash():
    geocircle = GeoCircle(Coordinate(0.0, 0.0), 1000, dt=default_test_datetime)

    c2 = GeoCircle(Coordinate(0.0, 0.0), 1000, dt=default_test_datetime)
    assert len({geocircle, c2}) == 1


def test_geocircle_repr():
    geocircle = GeoCircle(Coordinate(0.0, 0.0), 1000, dt=default_test_datetime)

    assert repr(geocircle) == '<GeoCircle at (0.0, 0.0); radius 1000.0 meters>'


def test_geocircle_bounds():
    actual = GeoCircle(Coordinate(0.0, 0.0), 1000).bounds
    expected = (-0.0089932, -0.0089932, 0.0089932, 0.0089932)
    for b1, b2 in zip(actual, expected):
        assert b1 == approx(b2, abs=1e-6)


def test_geocircle_to_polygon():
    geocircle = GeoCircle(Coordinate(0.0, 0.0), 1000, dt=default_test_datetime)

    assert geocircle.to_polygon() == GeoPolygon(geocircle.bounding_coords(), dt=default_test_datetime)


def test_geocircle_bounding_coords():
    geocircle = GeoCircle(Coordinate(0.0, 0.0), 1000, dt=default_test_datetime)

    circle = GeoCircle(Coordinate(0.0, 0.0), 1000)
    expected = [
        Coordinate(-0.0, 0.0089932),
        Coordinate(-0.0015617, 0.0088566),
        Coordinate(-0.0030759, 0.0084509),
        Coordinate(-0.0044966, 0.0077884),
        Coordinate(-0.0057807, 0.0068892)
    ]

    # Verify the first 5 coordinates
    for actual_coord, expected_coord in zip(circle.bounding_coords()[:5], expected):
        # We can now easily relax tolerance for complex geodesic calculations if needed
        assert_coordinates_equal(actual_coord, expected_coord, abs_tol=1e-6)


def test_geocircle_circumscribing_rectangle():
    geocircle = GeoCircle(Coordinate(0.0, 0.0), 1000, dt=default_test_datetime)
    actual = geocircle.circumscribing_rectangle()
    expected = GeoBox(
        Coordinate(-0.0089932, 0.0089932),
        Coordinate(0.0089932, -0.0089932),
        dt=default_test_datetime
    )
    assert_coordinates_equal(actual.nw_bound, expected.nw_bound)
    assert_coordinates_equal(actual.se_bound, expected.se_bound)


def test_geocircle_circumscribing_circle():
    geocircle = GeoCircle(Coordinate(0.0, 0.0), 1000, dt=default_test_datetime)

    assert geocircle.circumscribing_circle() == geocircle


def test_geocircle_centroid():
    geocircle = GeoCircle(Coordinate(0.0, 0.0), 1000, dt=default_test_datetime)

    assert geocircle.centroid == geocircle.center


def test_geocircle_copy():
    circle = GeoCircle(Coordinate(0., 1.), 500)
    circle_copy = circle.copy()

    # Assert equality but different pointer
    assert circle == circle_copy
    assert circle is not circle_copy


def test_geocircle_linear_rings():
    geocircle = GeoCircle(Coordinate(0.0, 0.0), 1000, dt=default_test_datetime)

    rings = geocircle.linear_rings()
    for actual_coord, expected_coord in zip(rings[0], geocircle.bounding_coords()):
        assert_coordinates_equal(actual_coord, expected_coord, abs_tol=1e-6)

    # Assert self-closing
    assert_coordinates_equal(rings[0][0], rings[0][-1])


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


def test_geoellipse_eq():
    geoellipse = GeoEllipse(Coordinate(0.0, 0.0), 1000, 500, 90, dt=default_test_datetime)

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


def test_geoellipse_hash():
    geoellipse = GeoEllipse(Coordinate(0.0, 0.0), 1000, 500, 90, dt=default_test_datetime)

    e2 = GeoEllipse(Coordinate(0.0, 0.0), 1000, 500, 90, dt=default_test_datetime)
    assert len({e2, geoellipse}) == 1


def test_geoellipse_repr():
    geoellipse = GeoEllipse(Coordinate(0.0, 0.0), 1000, 500, 90, dt=default_test_datetime)

    assert repr(geoellipse) == '<GeoEllipse at (0.0, 0.0); radius 1000.0/500.0; rotation 90.0>'


def test_geoellipse_bounds():
    actual = GeoEllipse(Coordinate(0.0, 0.0), 1000, 500, 45).bounds
    expected = (-0.0071098, -0.0071098, 0.0071098, 0.0071098)
    for b1, b2 in zip(actual, expected):
        approx(b1, b2, abs=1e-6)


def test_geoellipse_bounding_coords():
    geoellipse = GeoEllipse(Coordinate(0.0, 0.0), 1000, 500, 90)
    expected = [
        Coordinate(0.0089932, 0.0),
        Coordinate(0.0088586, 0.000775),
        Coordinate(0.0084813, 0.0014955),
        Coordinate(0.0079267, 0.002124),
        Coordinate(0.0072708, 0.0026464)
    ]
    for actual_coord, expected_coord in zip(geoellipse.bounding_coords()[:5], expected):
        assert_coordinates_equal(actual_coord, expected_coord, abs_tol=1e-6)

    # assert self-closing
    assert_coordinates_equal(geoellipse.bounding_coords()[0], geoellipse.bounding_coords()[-1])


def test_geoellipse_copy():
    ellipse = GeoEllipse(Coordinate(0., 1.), 500, 200, 90)
    ellipse_copy = ellipse.copy()

    # Assert equality but different pointer
    assert ellipse == ellipse_copy
    assert ellipse is not ellipse_copy


def test_geoellipse_linear_rings():
    geoellipse = GeoEllipse(Coordinate(0.0, 0.0), 1000, 500, 90, dt=default_test_datetime)

    rings = geoellipse.linear_rings()
    for actual_coord, expected_coord in zip(rings[0], geoellipse.bounding_coords()):
        assert_coordinates_equal(actual_coord, expected_coord, abs_tol=1e-6)

    # Assert self-closing
    assert_coordinates_equal(rings[0][0], rings[0][-1])


def test_geoellipse_to_polygon():
    geoellipse = GeoEllipse(Coordinate(0.0, 0.0), 1000, 500, 90, dt=default_test_datetime)

    assert geoellipse.to_polygon() == GeoPolygon(geoellipse.bounding_coords(), dt=default_test_datetime)


def test_geoellipse_to_geojson():
    geoellipse = GeoEllipse(Coordinate(0.0, 0.0), 1000, 500, 90, dt=default_test_datetime)

    shapely.geometry.shape(geoellipse.to_geojson(k=10, test_prop=2)['geometry'])


def test_geoellipse_circumscribing_rectangle():
    geoellipse = GeoEllipse(Coordinate(0.0, 0.0), 1000, 500, 90, dt=default_test_datetime)

    actual = geoellipse.circumscribing_rectangle()
    expected = GeoBox(
        Coordinate(-0.0089932, 0.0044966),
        Coordinate(0.0089932, -0.0044966),
        dt=default_test_datetime
    )
    assert_coordinates_equal(actual.nw_bound, expected.nw_bound)
    assert_coordinates_equal(actual.se_bound, expected.se_bound)


def test_geoellipse_circumscribing_circle():
    geoellipse = GeoEllipse(Coordinate(0.0, 0.0), 1000, 500, 90, dt=default_test_datetime)

    assert geoellipse.circumscribing_circle() == GeoCircle(
        geoellipse.centroid,
        geoellipse.semi_major,
        dt=default_test_datetime
    )


def test_geoellipse_centroid():
    geoellipse = GeoEllipse(Coordinate(0.0, 0.0), 1000, 500, 90, dt=default_test_datetime)

    assert geoellipse.centroid == geoellipse.center


def test_geoellipse_covariance_matrix():
    from numpy.testing import assert_allclose
    ellipse = GeoEllipse(Coordinate(0., 1.), 100, 50, 45)
    assert_allclose(ellipse.covariance_matrix(), np.array([[6250., 3750.], [3750., 6250.]]))

    ellipse = GeoEllipse(Coordinate(1., 2.), 100, 50, 90)
    assert_allclose(ellipse.covariance_matrix(), np.array([[10000., 0.], [0., 2500.]]))

    ellipse = GeoEllipse(Coordinate(1., 2.), 100, 50, 90)
    assert_allclose(
        ellipse.covariance_matrix(to_trigonometric_rotation=False),
        np.array([[2500., 0.], [0., 10000.]]),
        atol=1e-07
    )


def test_geoellipse_from_covariance_matrix():
    from numpy.testing import assert_allclose
    mean = Coordinate(1., 2.)

    cov = np.array([[6250., 3750.], [3750., 6250.]])
    expected = GeoEllipse(Coordinate(1., 2.), 100, 50, 45)
    actual = GeoEllipse.from_covariance_matrix(cov, mean)
    assert expected.centroid == actual.centroid
    for attr in ('semi_major', 'semi_minor', 'rotation'):
        assert_allclose(getattr(expected, attr), getattr(actual, attr))

    cov = np.array([[10000., 0.], [0., 2500.]])
    expected = GeoEllipse(Coordinate(1., 2.), 100, 50, 90)
    actual = GeoEllipse.from_covariance_matrix(cov, mean)
    assert expected.centroid == actual.centroid
    for attr in ('semi_major', 'semi_minor', 'rotation'):
        assert_allclose(getattr(expected, attr), getattr(actual, attr))

    cov = np.array([[2500., 0.], [0., 10000.]])
    expected = GeoEllipse(Coordinate(1., 2.), 100, 50, 90)
    actual = GeoEllipse.from_covariance_matrix(cov, mean, from_trigonometric_rotation=False)
    assert expected.centroid == actual.centroid
    for attr in ('semi_major', 'semi_minor', 'rotation'):
        assert_allclose(getattr(expected, attr), getattr(actual, attr))


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
    assert repr(geowedge) == '<GeoRing at (0.0, 0.0); radii 500.0/1000.0; 90.0-180.0 degrees>'
    assert repr(georing) == '<GeoRing at (0.0, 0.0); radii 500.0/1000.0>'


def test_georing_bounds():
    ring = GeoRing(Coordinate(0., 0.), 1000, 5000)
    expected = (-0.0449661, -0.0449661, 0.0449661, 0.0449661)
    for b1, b2 in zip(ring.bounds, expected):
        assert b1 == approx(b2, abs=1e-6)

    wedge = GeoRing(Coordinate(0., 0.), 1000, 5000, 90, 180)
    expected = (0., -0.0449661, 0.0449661, 0.)
    for b1, b2 in zip(wedge.bounds, expected):
        assert b1 == approx(b2, abs=1e-6)


def test_georing_bounding_coords(geowedge, georing):
    wedge = GeoRing(Coordinate(0.0, 0.0), 500, 1000, 90, 180)
    expected = [
        Coordinate(0.0, -0.0089932),
        Coordinate(0.0014068, -0.0088825),
        Coordinate(0.0027791, -0.0085531),
        Coordinate(0.0040828, -0.008013),
        Coordinate(0.0052861, -0.0072757),
    ]
    for c1, c2 in zip(wedge.bounding_coords(), expected):
        assert_coordinates_equal(c1, c2)

    # Assert self-closing
    assert_coordinates_equal(wedge.bounding_coords()[0], wedge.bounding_coords()[-1])

    ring = GeoRing(Coordinate(0.0, 0.0), 500, 1000)
    expected = [
        Coordinate(-0.0, 0.0089932),
        Coordinate(-0.0015617, 0.0088566),
        Coordinate(-0.0030759, 0.0084509),
        Coordinate(-0.0044966, 0.0077884),
        Coordinate(-0.0057807, 0.0068892),
    ]
    for c1, c2 in zip(ring.bounding_coords(), expected):
        assert_coordinates_equal(c1, c2)

    # assert self-closing
    assert_coordinates_equal(ring.bounding_coords()[0], ring.bounding_coords()[-1])


def test_georing_copy():
    ring = GeoRing(Coordinate(0., 1.), 500, 200)
    ring_copy = ring.copy()

    # Assert equality but different pointer
    assert ring == ring_copy
    assert ring is not ring_copy


def test_georing_to_geojson(georing):
    shapely.geometry.shape(georing.to_geojson()['geometry'])


def test_georing_circumscribing_rectangle(georing, geowedge):

    max_lon, _ = destination_point(georing.centroid, 90, 1000).to_float()
    min_lon, _ = destination_point(georing.centroid, -90, 1000).to_float()
    _, max_lat = destination_point(georing.centroid, 0, 1000).to_float()
    _, min_lat = destination_point(georing.centroid, 180, 1000).to_float()

    actual = georing.circumscribing_rectangle()
    expected = GeoBox(
        Coordinate(min_lon, max_lat),
        Coordinate(max_lon, min_lat),
        dt=default_test_datetime
    )
    assert_coordinates_equal(actual.nw_bound, expected.nw_bound)
    assert_coordinates_equal(actual.se_bound, expected.se_bound)

    actual = geowedge.circumscribing_rectangle()
    expected = GeoBox(
        geowedge.center,
        Coordinate(max_lon, min_lat),
        dt=default_test_datetime
    )
    assert_coordinates_equal(actual.nw_bound, expected.nw_bound)
    assert_coordinates_equal(actual.se_bound, expected.se_bound)


def test_georing_circumscribing_circle():
    ring = GeoRing(Coordinate(0.0, 0.0), 500, 1000)
    actual = ring.circumscribing_circle()
    expected = GeoCircle(
        ring.centroid,
        ring.outer_radius
    )
    assert_coordinates_equal(actual.centroid, expected.centroid)
    assert actual.radius == approx(expected.radius, abs=1e-6)

    wedge = GeoRing(Coordinate(0.0, 0.0), 500, 1000, 90, 180)
    circ = wedge.circumscribing_circle()
    received_centroid = Coordinate(
        round_half_up(circ.centroid.longitude, 8),
        round_half_up(circ.centroid.latitude, 8),
    )
    assert_coordinates_equal(received_centroid, Coordinate(0.00444382, -0.00444382))
    assert circ.radius == approx(707.1555054, abs=1e-6)


def test_georing_centroid():
    ring = GeoRing(Coordinate(0.0, 0.0), 500, 1000)
    assert ring.centroid == Coordinate(0.0, 0.0)

    wedge = GeoRing(Coordinate(0.0, 0.0), 500, 1000, 90, 180)
    received = Coordinate(
        round_half_up(wedge.centroid.longitude, 8),
        round_half_up(wedge.centroid.latitude, 8),
    )
    assert received == Coordinate(0.00444382, -0.00444382)


def test_georing_linear_rings(georing, geowedge):
    georing = GeoRing(Coordinate(0.0, 0.0), 500, 1000)
    rings = georing.linear_rings()
    assert len(rings) == 2  # should have outer and inner shell

    expected = [
        Coordinate(-0.0, 0.0089932),
        Coordinate(-0.0015617, 0.0088566),
        Coordinate(-0.0030759, 0.0084509),
        Coordinate(-0.0044966, 0.0077884),
        Coordinate(-0.0057807, 0.0068892)
    ]
    for c1, c2 in zip(rings[0][:5], expected):
        assert_coordinates_equal(c1, c2)

    # Assert self-closing
    assert_coordinates_equal(rings[0][0], rings[0][-1])
    assert_coordinates_equal(rings[1][0], rings[1][-1])

    rings = geowedge.linear_rings()
    assert len(rings) == 1
    assert_coordinates_equal(rings[0][0], rings[0][-1])


def test_georing_to_wkt():
    # --- Case 1: Full Ring (Circle with a hole) ---
    ring = GeoRing(Coordinate(0.0, 0.0), 500, 1000)
    wkt_str = ring.to_wkt()

    parsed_poly = GeoPolygon.from_wkt(wkt_str)

    expected_outer = GeoCircle(Coordinate(0.0, 0.0), 1000).bounding_coords()
    expected_inner = GeoCircle(Coordinate(0.0, 0.0), 500).bounding_coords()
    expected_poly = GeoPolygon(expected_outer, holes=[GeoPolygon(expected_inner)])

    assert_geopolygons_equal(parsed_poly, expected_poly)

    # --- Case 2: Wedge (Simple Polygon) ---
    wedge = GeoRing(Coordinate(0.0, 0.0), 500, 1000, 90, 180)
    wkt_str_wedge = wedge.to_wkt()
    parsed_wedge = GeoPolygon.from_wkt(wkt_str_wedge)

    # For a wedge, the WKT is just the standard polygon representation
    # so we can simply compare it against the direct polygon conversion.
    expected_wedge = wedge.to_polygon()

    assert_geopolygons_equal(parsed_wedge, expected_wedge)

def test_georing_to_polygon():
    georing = GeoRing(Coordinate(0.0, 0.0), 500, 1000, dt=default_test_datetime)

    # Because its not a wedge, should become a polygon with a hole
    rings = georing.linear_rings()
    assert georing.to_polygon() == GeoPolygon(
        rings[0],
        holes=[GeoPolygon(rings[1])],
        dt=default_test_datetime
    )

    # Is a wedge, so just a normal polygon
    geowedge = GeoRing(Coordinate(0.0, 0.0), 500, 1000, 90, 180, dt=default_test_datetime)
    rings = geowedge.linear_rings()
    assert geowedge.to_polygon() == GeoPolygon(
        rings[0],
        dt=default_test_datetime
    )


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


def test_geolinestring_bounds():
    ls = GeoLineString([Coordinate(0.0, 0.0), Coordinate(1.0, 0.0), Coordinate(1.0, 1.0)])
    assert ls.bounds == (
        0., 0., 1., 1.
    )


def test_geolinestring_circumscribing_rectangle():
    ls = GeoLineString([Coordinate(0.0, 0.0), Coordinate(1.0, 0.0), Coordinate(1.0, 1.0)])
    assert ls.circumscribing_rectangle() == GeoBox(
        Coordinate(0., 1.),
        Coordinate(1., 0.)
    )


def test_geolinestring_contains():
    ls = GeoLineString([Coordinate(0., 0.), Coordinate(1., 1.), Coordinate(2., 2.)], dt=datetime(2020, 1, 1))

    ls2 = GeoCircle(Coordinate(0., 0.), 500)
    assert not ls.contains(ls2)

    ls2 = GeoLineString([Coordinate(0., 0.), Coordinate(1., 1.)])
    assert ls.contains(ls2)

    ls2 = GeoLineString([Coordinate(0., 0.), Coordinate(1., 1.)], dt=datetime(2020, 1, 2))
    assert not ls.contains(ls2)

    ls2 = GeoLineString([Coordinate(0., 1.), Coordinate(0., 0.), Coordinate(1., 1.)])
    assert not ls.contains(ls2)

    point = GeoPoint(Coordinate(0., 0.))
    assert ls.contains(point)

    point2 = GeoPoint(Coordinate(1., 0.))
    assert not ls.contains(point2)

    assert ls.contains_shape(MultiGeoPoint([GeoPoint(Coordinate(0., 0.))]))
    assert not ls.contains_shape(MultiGeoPoint([GeoPoint(Coordinate(5., 0.))]))


def test_geolinestring_copy():
    linestring = GeoLineString([Coordinate(0., 1.), Coordinate(0., 1.)])
    linestring_copy = linestring.copy()

    # Assert equality but different pointer
    assert linestring == linestring_copy
    assert linestring is not linestring_copy


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

    gls = gls['geometry']
    expected = GeoLineString(
        [Coordinate(0.0, 0.0), Coordinate(1.0, 1.5), Coordinate(2.0, 2.0)],
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


def test_geolinestring_intersects_shape():
    ls = GeoLineString([Coordinate(0.0, 0.0), Coordinate(1.0, 1.0)])
    circle = GeoCircle(Coordinate(0.0, 0.0), 5_000)
    assert ls.intersects_shape(circle)

    ls = GeoLineString([Coordinate(0.0, 0.0), Coordinate(1.0, 1.0)])
    circle = GeoCircle(Coordinate(0.0, 0.0), 5_000)
    assert ls.intersects_shape(circle)

    ls = GeoLineString([Coordinate(0.0, 0.0), Coordinate(1.0, 1.0)])
    assert ls.intersects_shape(ls)

    # Contains point
    assert ls.intersects_shape(GeoPoint(Coordinate(0., 0.)))

    assert ls.intersects_shape(MultiGeoPoint([GeoPoint(Coordinate(0., 0.))]))
    assert not ls.intersects_shape(MultiGeoPoint([GeoPoint(Coordinate(5., 0.))]))


def test_geolinestring_to_geojson(geolinestring):

    assert geolinestring.to_geojson(properties={'test_prop': 2}, test_kwarg=1) == {
        'type': 'Feature',
        'geometry': {
            'type': 'LineString',
            'coordinates': [list(x.to_float()) for x in geolinestring.vertices],
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
    assert geolinestring.to_wkt() == 'LINESTRING(0 0,1 0,1 1)'

    wkt.loads(geolinestring.to_wkt())


def test_geolinestring_to_polygon(geolinestring):
    ls = GeoLineString(
        [Coordinate(0.0, 0.0), Coordinate(1.0, 0.0), Coordinate(1.0, 1.0)],
        dt=default_test_datetime
    )
    assert ls.to_polygon() == GeoPolygon(
        [Coordinate(0.0, 0.0), Coordinate(1.0, 0.0), Coordinate(1.0, 1.0), Coordinate(0.0, 0.0)],
        dt=default_test_datetime
    )


def test_geopoint_contains_dunder(geopoint):
    assert geopoint in geopoint


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


def test_geopoint_bounds():
    point = GeoPoint(Coordinate(0., 0.))
    assert point.bounds == (0., 0., 0., 0.)


def test_geopoint_copy():
    point = GeoPoint(Coordinate(0., 1.))
    point_copy = point.copy()

    # Assert equality but different pointer
    assert point == point_copy
    assert point is not point_copy


def test_geopoint_contains():
    assert GeoPoint(Coordinate(0., 0.)).contains(GeoPoint(Coordinate(0., 0.)))
    assert not GeoPoint(Coordinate(0., 0.)).contains(GeoPoint(Coordinate(1., 1.)))


def test_geopoint_contains_coordinate():
    assert GeoPoint(Coordinate(0., 0.)).contains_coordinate(Coordinate(0., 0.))
    assert not GeoPoint(Coordinate(0., 0.)).contains_coordinate(Coordinate(1., 0.))


def test_geopoint_contains_shape():
    assert GeoPoint(Coordinate(0., 0.)).contains_shape(GeoPoint(Coordinate(0., 0.)))
    assert GeoPoint(Coordinate(0., 0.)).contains_shape(MultiGeoPoint([GeoPoint(Coordinate(0., 0.))]))
    assert not GeoPoint(Coordinate(0., 0.)).contains_shape(GeoCircle(Coordinate(0., 0.), 10))
    assert not GeoPoint(Coordinate(0., 0.)).contains_shape(MultiGeoPoint([GeoPoint(Coordinate(1., 0.))]))


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

    # Only geo interface
    gpoint = gpoint['geometry']
    expected = GeoPoint(
        Coordinate(1.0, 0.0),
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


def test_geopoint_intersects_shape():
    point = GeoPoint(Coordinate(0., 0.))
    assert point.intersects_shape(GeoCircle(Coordinate(0., 0.), 500))
    assert point.intersects_shape(point)

    assert not point.intersects_shape(GeoCircle(Coordinate(1., 0.), 500))
    assert not point.intersects_shape(GeoPoint(Coordinate(0.001, 0.001)))


def test_geopoint_to_geojson():
    point = GeoPoint(Coordinate(1., 0.), dt=default_test_datetime)
    assert point.to_geojson(properties={'test_prop': 2}, test_kwarg=1) == {
        'type': 'Feature',
        'geometry': {
            'type': 'Point',
            'coordinates': [1.0, 0.0],
        },
        'properties': {
            'test_prop': 2,
            'datetime_start': default_test_datetime.isoformat(),
            'datetime_end': default_test_datetime.isoformat(),
        },
        'test_kwarg': 1
    }


def test_geopoint_to_shapely(geopoint):
    assert geopoint.to_shapely() == shapely.Point(0.0, 0.0)


def test_geopoint_centroid(geopoint):
    assert geopoint.centroid == geopoint.coordinate


def test_geopoint_from_wkt():
    wkt_str = 'POINT (1.0 1.0)'
    assert GeoPoint.from_wkt(wkt_str) == GeoPoint(Coordinate(1.0, 1.0))

    wkt_str = 'POINT(1.0 1.0)'
    assert GeoPoint.from_wkt(wkt_str) == GeoPoint(Coordinate(1.0, 1.0))

    wkt_str = 'POINT(-1.0 -1.0)'
    assert GeoPoint.from_wkt(wkt_str) == GeoPoint(Coordinate(-1.0, -1.0))

    with pytest.raises(ValueError):
        _ = GeoPoint.from_wkt('NOT WKT')


def test_geopoint_to_wkt(geopoint):
    assert geopoint.to_wkt() == 'POINT(0 0)'


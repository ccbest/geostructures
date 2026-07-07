
from datetime import datetime

import pytest
import shapely

from geostructures import GeoBox, GeoCircle, GeoPolygon
from geostructures.coordinates import Coordinate
from geostructures.utils.functions import round_half_up

from tests.functions import (
    assert_shape_equivalence, default_test_datetime,
    geojson_round_trip, shapely_round_trip, wkt_round_trip,
)


def test_geopolygon_eq():
    poly = GeoPolygon(
        [
            Coordinate(0.0, 0.0), Coordinate(0.0, 1.0), Coordinate(1.0, 1.0),
            Coordinate(1.0, 0.0), Coordinate(0.0, 0.0)
        ],
        dt=default_test_datetime
    )
    p2 = GeoPolygon(
        [
            Coordinate(0.0, 0.0), Coordinate(0.0, 1.0), Coordinate(1.0, 1.0),
            Coordinate(1.0, 0.0), Coordinate(0.0, 0.0)
        ],
        dt=default_test_datetime
    )
    assert poly == p2

    # Same vertices, but rotated
    p2 = GeoPolygon(
        [
            Coordinate(0.0, 1.0), Coordinate(1.0, 1.0), Coordinate(1.0, 0.0),
            Coordinate(0.0, 0.0), Coordinate(0.0, 1.0),
        ],
        dt=default_test_datetime
    )
    assert poly == p2

    # Same vertices, but reversed
    p2 = GeoPolygon(
        poly.outline[::-1],
        dt=default_test_datetime
    )
    assert poly == p2

    # Different vertices - different lengths
    p2 = GeoPolygon(
        [
            Coordinate(1.0, 1.0), Coordinate(0.0, 1.0), Coordinate(1.0, 1.0),
            Coordinate(1.0, 0.0), Coordinate(0.0, 0.0)
        ],
        dt=default_test_datetime
    )
    assert poly != p2

    # Different vertices - same lengths
    p2 = GeoPolygon(
        [
            Coordinate(1.0, 1.0), Coordinate(0.0, 1.0),
            Coordinate(1.0, 0.0), Coordinate(0.0, 0.0)
        ],
        dt=default_test_datetime
    )
    assert poly != p2

    # Different datetime
    p2 = GeoPolygon(
        [
            Coordinate(0.0, 0.0), Coordinate(0.0, 1.0), Coordinate(1.0, 1.0),
            Coordinate(1.0, 0.0), Coordinate(0.0, 0.0)
        ],
        dt=datetime(1970, 1, 1, 1, 0)
    )
    assert poly != p2

    # Not a geopolygon
    assert 'test' != poly

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


def test_geopolygon_hash():
    p1 = GeoPolygon(
        [
            Coordinate(0.0, 0.0), Coordinate(0.0, 1.0), Coordinate(1.0, 1.0),
            Coordinate(1.0, 0.0), Coordinate(0.0, 0.0)
        ],
        dt=default_test_datetime
    )
    p2 = GeoPolygon(
        [
            Coordinate(0.0, 0.0), Coordinate(0.0, 1.0), Coordinate(1.0, 1.0),
            Coordinate(1.0, 0.0), Coordinate(0.0, 0.0)
        ],
        dt=default_test_datetime
    )
    assert len({p1, p2}) == 1


def test_geopolygon_hash_matches_equality():
    # __eq__ is rotation/orientation-invariant, so __hash__ must be as well
    p1 = GeoPolygon([
        Coordinate(0., 0.), Coordinate(1., 0.), Coordinate(1., 1.),
        Coordinate(0., 1.), Coordinate(0., 0.)
    ])
    p2 = GeoPolygon([
        Coordinate(1., 1.), Coordinate(0., 1.), Coordinate(0., 0.),
        Coordinate(1., 0.), Coordinate(1., 1.)
    ])
    assert p1 == p2
    assert hash(p1) == hash(p2)
    assert len({p1, p2}) == 1


def test_geopolygon_repr():
    poly = GeoPolygon([
        Coordinate(0.0, 0.0), Coordinate(0.0, 1.0), Coordinate(1.0, 1.0),
        Coordinate(1.0, 0.0), Coordinate(0.0, 0.0)
    ])
    assert repr(poly) == '<GeoPolygon of 4 coordinates>'


def test_geopolygon_bounds():
    poly = GeoPolygon(
        [
            Coordinate(0.0, 0.0), Coordinate(0.0, 1.0), Coordinate(1.0, 1.0),
            Coordinate(1.0, 0.0), Coordinate(0.0, 0.0)
        ],
    )
    assert poly.bounds == (0., 0., 1., 1.)


def test_geopolygon_bounding_coords():
    poly = GeoPolygon([
        Coordinate(0.0, 0.0), Coordinate(0.0, 1.0), Coordinate(1.0, 1.0),
        Coordinate(1.0, 0.0), Coordinate(0.0, 0.0)
    ])
    assert poly.outline == poly.bounding_coords()

    # assert self-closing
    assert poly.bounding_coords()[0] == poly.bounding_coords()[-1]


def test_geopolygon_copy():
    poly = GeoPolygon(
        [Coordinate(0.0, 0.0), Coordinate(1.0, 1.0), Coordinate(2.0, 0.0), Coordinate(0.0, 0.0)],
        holes=[GeoPolygon([
            Coordinate(0.25, 0.25), Coordinate(0.5, 0.5), Coordinate(1.0, 0.25), Coordinate(0.25, 0.25)
        ])],
        properties={'example': 'prop'}
    )
    poly_copy = poly.copy()

    # Assert equality but different pointer
    assert poly == poly_copy
    assert poly is not poly_copy


def test_geopolygon_serialization_round_trips():
    poly = GeoPolygon(
        [Coordinate(0.0, 0.0), Coordinate(1.0, 1.0), Coordinate(2.0, 0.0), Coordinate(0.0, 0.0)],
        holes=[GeoPolygon([
            Coordinate(0.25, 0.25), Coordinate(0.5, 0.5), Coordinate(1.0, 0.25), Coordinate(0.25, 0.25)
        ])],
        dt=default_test_datetime,
    )
    wkt_round_trip(poly)
    geojson_round_trip(poly)
    shapely_round_trip(poly)


def test_geopolygon_to_wkt():
    poly = GeoPolygon(
        [
            Coordinate(0.0, 0.0), Coordinate(1.0, 0.0), Coordinate(1.0, 1.0),
            Coordinate(0.0, 1.0), Coordinate(0.0, 0.0)
        ],
        holes=[GeoPolygon([
            Coordinate(0.25, 0.25), Coordinate(0.75, 0.25), Coordinate(0.75, 0.75),
            Coordinate(0.25, 0.75), Coordinate(0.25, 0.25)
        ])]
    )
    assert poly.to_wkt() == \
        'POLYGON((0 0,1 0,1 1,0 1,0 0), (0.25 0.25,0.25 0.75,0.75 0.75,0.75 0.25,0.25 0.25))'


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
        holes=[GeoPolygon([
            Coordinate(0.25, 0.25), Coordinate(0.5, 0.5), Coordinate(1.0, 0.25), Coordinate(0.25, 0.25)
        ])],
        properties={'example': 'prop'}
    )
    assert GeoPolygon.from_geojson(gjson) == expected

    # Only geo interface
    gjson = gjson['geometry']
    expected = GeoPolygon(
        [Coordinate(0.0, 0.0), Coordinate(1.0, 1.0), Coordinate(2.0, 0.0), Coordinate(0.0, 0.0)],
        holes=[GeoPolygon([
            Coordinate(0.25, 0.25), Coordinate(0.5, 0.5), Coordinate(1.0, 0.25), Coordinate(0.25, 0.25)
        ])],
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
            Coordinate(-0.0006950065, -0.0052490565),
            Coordinate(0.0012379757, -0.0012078765),
            Coordinate(-0.0007774661, 0.0022317121),
            Coordinate(-0.0047256093, 0.0016301653),
            Coordinate(-0.0066584556, -0.0024107268),
            Coordinate(-0.0046432944, -0.0058503599),
            Coordinate(-0.0006950065, -0.0052490565)
        ]
    )
    assert_shape_equivalence(
        GeoPolygon.from_h3_geohash(geohash),
        expected,
        5
    )


def test_geopolygon_from_niemeyer_geohash():
    geohash = "3fffffff"
    # A base-16, length-8 geohash encodes 16 bits of longitude and 16 bits of
    # latitude, so the cell spans 360/2**16 degrees of longitude but only
    # 180/2**16 degrees of latitude
    expected = GeoBox(
        Coordinate(-0.0054931640625, 0.0),
        Coordinate(0.0, -0.00274658203125)
    ).to_polygon()
    assert_shape_equivalence(
        GeoPolygon.from_niemeyer_geohash(geohash, 16),
        expected,
        5
    )


def test_geopolygon_circumscribing_circle():
    poly = GeoPolygon(
        [
            Coordinate(0.0, 0.0), Coordinate(0.0, 1.0), Coordinate(1.0, 1.0),
            Coordinate(1.0, 0.0), Coordinate(0.0, 0.0)
        ],
    )
    gc = poly.circumscribing_circle()
    assert round_half_up(gc.center.latitude, 4) == 0.5
    assert round_half_up(gc.center.longitude, 4) == 0.5
    assert round_half_up(gc.radius, 0) == 78625


def test_geopolygon_circumscribing_rectangle():
    poly = GeoPolygon(
        [
            Coordinate(0.0, 0.0), Coordinate(0.0, 1.0), Coordinate(1.0, 1.0),
            Coordinate(1.0, 0.0), Coordinate(0.0, 0.0)
        ],
        dt=default_test_datetime
    )
    assert poly.circumscribing_rectangle() == GeoBox(
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


def test_geopolygon_contains_coordinate_vertex_ray():
    # The eastward test ray passes through the vertex at (5, 0); this used to
    # be misread as "point on boundary" and return False for interior points
    diamond = GeoPolygon([
        Coordinate(0., -1.), Coordinate(5., 0.), Coordinate(0., 1.),
        Coordinate(-5., 0.), Coordinate(0., -1.)
    ])
    assert diamond.contains_coordinate(Coordinate(0., 0.))
    assert diamond.contains_coordinate(Coordinate(-2., 0.))
    assert not diamond.contains_coordinate(Coordinate(6., 0.))
    assert not diamond.contains_coordinate(Coordinate(-6., 0.))

    # Points exactly on the boundary are not contained
    assert not diamond.contains_coordinate(Coordinate(5., 0.))
    assert not diamond.contains_coordinate(Coordinate(2.5, 0.5))


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


def test_geopolygon_to_polygon():
    poly = GeoPolygon(
        [
            Coordinate(0.0, 0.0), Coordinate(0.0, 1.0), Coordinate(1.0, 1.0),
            Coordinate(1.0, 0.0), Coordinate(0.0, 0.0)
        ],
        dt=default_test_datetime
    )
    assert poly.to_polygon() == poly


def test_geopolygon_to_shapely():
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


def test_geopolygon_centroid_with_z():
    poly = GeoPolygon([
        Coordinate(0., 0., z=5.), Coordinate(1., 0., z=5.),
        Coordinate(1., 1., z=5.), Coordinate(0., 0., z=5.)
    ])
    assert poly.centroid.z == 5.


def test_geopolygon_point_on_horizontal_edge():
    square = GeoPolygon([
        Coordinate(0., 0.), Coordinate(1., 0.), Coordinate(1., 1.),
        Coordinate(0., 1.), Coordinate(0., 0.)
    ])
    # Point on the horizontal bottom edge is a boundary, not contained
    assert not square.contains_coordinate(Coordinate(0.5, 0.))


def test_point_in_polygon_antimeridian_edges():
    # Call the ray-caster directly; contains_coordinate's bbox pre-filter
    # does not handle antimeridian-crossing polygons
    outline = [
        Coordinate(179., 0.), Coordinate(-179., 0.), Coordinate(-179., 1.),
        Coordinate(179., 1.), Coordinate(179., 0.)
    ]
    assert GeoPolygon._point_in_polygon(Coordinate(179.5, 0.5), outline)
    assert GeoPolygon._point_in_polygon(Coordinate(-179.5, 0.5), outline)
    assert not GeoPolygon._point_in_polygon(Coordinate(178., 0.5), outline)

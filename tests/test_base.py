
from datetime import datetime, timedelta, timezone

import pytest
import shapely

from geostructures import *
from geostructures.multistructures import *
from geostructures.time import TimeInterval


def test_geoshape_buffer_dt():
    # Base Case
    point = GeoPoint(Coordinate('0.0', '0.0'), dt=TimeInterval(datetime(2020, 1, 1, 12), datetime(2020, 1, 3, 12)))
    point2 = point.buffer_dt(timedelta(hours=1))
    assert point2 == GeoPoint(
        Coordinate('0.0', '0.0'),
        dt=TimeInterval(datetime(2020, 1, 1, 11), datetime(2020, 1, 3, 13))
    )

    # In place
    point.buffer_dt(timedelta(hours=2), inplace=True)
    assert point == GeoPoint(
        Coordinate('0.0', '0.0'),
        dt=TimeInterval(datetime(2020, 1, 1, 10), datetime(2020, 1, 3, 14))
    )

    point = GeoPoint(Coordinate('0.0', '0.0'))
    with pytest.raises(ValueError):
        point.buffer_dt(timedelta(hours=1))


def test_geoshape_contains_dunder():
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


def test_geoshape_contains():
    # Base Case
    circle_outer = GeoCircle(Coordinate(0., 0.), 5_000, dt=TimeInterval(datetime(2020, 1, 2), datetime(2020, 1, 3)))
    circle_inner = GeoCircle(Coordinate(0., 0.), 2_000, dt=datetime(2020, 1, 2, 12))
    assert circle_outer.contains(circle_inner)

    # Time bounding
    circle_outer = GeoCircle(Coordinate(0., 0.), 5_000, dt=datetime(2020, 1, 2))
    circle_inner = GeoCircle(Coordinate(0., 0.), 2_000, dt=datetime(2020, 1, 1))
    assert not circle_outer.contains(circle_inner)

    # dt not defined
    circle_outer = GeoCircle(Coordinate(0., 0.), 5_000)
    circle_inner = GeoCircle(Coordinate(0., 0.), 2_000)
    assert circle_outer.contains(circle_inner)


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


def test_geoshape_intersects_time():
    geopoint = GeoPoint(Coordinate('0.0', '0.0'), dt=datetime(2020, 1, 1, 1))
    assert geopoint.intersects_time(datetime(2020, 1, 1, 1))
    assert not geopoint.intersects_time(datetime(2020, 1, 1, 1, 1))
    assert geopoint.intersects_time(TimeInterval(datetime(2020, 1, 1, 1), datetime(2020, 1, 1, 1, 1)))

    geopoint = GeoPoint(Coordinate('0.0', '0.0'), dt=TimeInterval(datetime(2020, 1, 1, 12), datetime(2020, 1, 3, 12)))
    assert geopoint.intersects_time(datetime(2020, 1, 2))
    assert not geopoint.intersects_time(datetime(2020, 1, 4, 12))
    assert geopoint.intersects_time(TimeInterval(datetime(2020, 1, 1, 14),datetime(2020, 1, 1, 16)))

    geopoint = GeoPoint(Coordinate('0.0', '0.0'), dt=None)
    assert not geopoint.intersects_time(datetime(2020, 1, 4, 12))
    assert not geopoint.intersects_time(TimeInterval(datetime(2020, 1, 1, 14), datetime(2020, 1, 1, 16)))


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


def test_geoshape_strip_dt():
    point = GeoPoint(Coordinate('0.0', '0.0'), dt=datetime(2020, 1, 1, 12))
    expected = GeoPoint(Coordinate('0.0', '0.0'))
    assert point.strip_dt() == expected


def test_shapelike_contains_shape():
    # Base case
    circle_outer = GeoCircle(Coordinate(0., 0.), 5_000)
    circle_inner = GeoCircle(Coordinate(0., 0.), 2_000)
    assert circle_outer.contains_shape(circle_inner)
    assert not circle_inner.contains_shape(circle_outer)

    # Intersecting
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


def test_shapelike_edges():
    polygon = GeoPolygon(
        [
            Coordinate(1.0, 0.0), Coordinate(1.0, 1.0),
            Coordinate(0.0, 0.5), Coordinate(1.0, 0.0)
        ],
        holes=[
            GeoPolygon([
                Coordinate(0.4, 0.6), Coordinate(0.5, 0.4),
                Coordinate(0.6, 0.6), Coordinate(0.4, 0.6),
            ])
        ]
    )
    assert polygon.edges() == [
        [
            (Coordinate(1.0, 0.0), Coordinate(1.0, 1.0)),
            (Coordinate(1.0, 1.0), Coordinate(0.0, 0.5)),
            (Coordinate(0.0, 0.5), Coordinate(1.0, 0.0)),
        ],
        [
            (Coordinate(0.4, 0.6), Coordinate(0.6, 0.6)),
            (Coordinate(0.6, 0.6), Coordinate(0.5, 0.4)),
            (Coordinate(0.5, 0.4), Coordinate(0.4, 0.6)),
        ]
    ]


def test_shapelike_intersects_shape():
    circle1 = GeoCircle(Coordinate(0.0, 0.0), 5_000)
    circle2 = GeoCircle(Coordinate(0.0899322, 0.0), 5_000)  # Exactly 10km to the right
    # Exactly one point where shapes intersect (boundary)
    assert circle1.intersects_shape(circle2)
    assert circle2.intersects_shape(circle1)

    circle1 = GeoCircle(Coordinate(0.0, 0.0), 5_000)
    circle2 = GeoCircle(Coordinate(0.0899321, 0.0), 5_000)  # Nudged just barely to the left
    assert circle1.intersects_shape(circle2)
    assert circle2.intersects_shape(circle1)

    circle1 = GeoCircle(Coordinate(0.0, 0.0), 5_000)
    circle2 = GeoCircle(Coordinate(0.0899323, 0.0), 5_000)  # Nudged just barely to the right
    assert not circle1.intersects_shape(circle2)
    assert not circle2.intersects_shape(circle1)

    circle1 = GeoCircle(Coordinate(0.0, 0.0), 5_000)
    circle2 = GeoCircle(Coordinate(0.0, 0.0), 2_000)  # Fully contained
    assert circle1.intersects_shape(circle2)
    assert circle2.intersects_shape(circle1)

    # points
    point = GeoPoint(Coordinate(0., 0.))
    assert circle1.intersects_shape(point)
    assert point.intersects_shape(circle1)


def test_multigeobase_contains():
    mls = MultiGeoLineString(
        [
            GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5), Coordinate(1., 0.)]),
            GeoLineString([Coordinate(1., 1.), Coordinate(0.5, 0.5), Coordinate(0., 0.)]),
        ],
        dt=TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 2))
    )
    assert Coordinate(0.5, 0.5) in mls
    assert GeoPoint(Coordinate(0.5, 0.5)) in mls
    assert GeoPoint(Coordinate(0.5, 0.5), dt=datetime(2020, 1, 1, 1))


def test_multigeobase_hash():
    mls_1 = MultiGeoLineString(
        [
            GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5), Coordinate(1., 0.)]),
            GeoLineString([Coordinate(1., 1.), Coordinate(0.5, 0.5), Coordinate(0., 0.)]),
        ],
        dt=TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 2))
    )
    mls_2 = MultiGeoLineString(
        [
            GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5), Coordinate(1., 0.)]),
            GeoLineString([Coordinate(1., 1.), Coordinate(0.5, 0.5), Coordinate(0., 0.)]),
        ],
        dt=TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 2))
    )
    assert hash(mls_1) == hash(mls_2)

    mls_2 = MultiGeoLineString(
        [
            GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5), Coordinate(1., 0.)]),
            GeoLineString([Coordinate(1., 1.), Coordinate(0.5, 0.5), Coordinate(0., 0.)]),
        ],
        dt=TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 3))
    )
    assert hash(mls_1) != hash(mls_2)


def test_multigeobase_bounds():
    mls = MultiGeoLineString(
        [
            GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5), Coordinate(1., 0.)]),
            GeoLineString([Coordinate(2., 1.), Coordinate(0.5, 0.5), Coordinate(0., 0.)]),
        ],
    )
    assert mls.bounds == ((0., 2.), (0., 1.))


def test_multigeobase_contains_coordinate():
    mls = MultiGeoLineString(
        [
            GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5), Coordinate(1., 0.)]),
            GeoLineString([Coordinate(1., 1.), Coordinate(0.5, 0.5), Coordinate(0., 0.)]),
        ],
    )
    assert mls.contains_coordinate(Coordinate(0.5, 0.5))
    assert not mls.contains_coordinate(Coordinate(2.0, 2.0))


def test_multigeobase_contains_shape():
    mls = MultiGeoLineString(
        [
            GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5), Coordinate(1., 0.)]),
            GeoLineString([Coordinate(1., 1.), Coordinate(0.5, 0.5), Coordinate(0., 0.)]),
        ],
    )
    assert mls.contains_shape(
        MultiGeoLineString(
            [
                GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5)]),
                GeoLineString([Coordinate(1., 1.), Coordinate(0.5, 0.5)]),
            ],
        )
    )
    assert mls.contains_shape(
        MultiGeoPoint([GeoPoint(Coordinate(0., 1.)), GeoPoint(Coordinate(1., 1.))])
    )

    assert not mls.contains_shape(GeoCircle(Coordinate(0., 0.), 500))
    assert not mls.contains_shape(GeoPoint(Coordinate(2., 2.)))


def test_multigeobase_copy():
    mls_1 = MultiGeoLineString(
        [
            GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5), Coordinate(1., 0.)]),
            GeoLineString([Coordinate(1., 1.), Coordinate(0.5, 0.5), Coordinate(0., 0.)]),
        ],
    )
    mls_2 = mls_1.copy()
    assert mls_1 == mls_2

    mls_2.geoshapes.pop(0)
    assert mls_1 != mls_2


def test_multigeobase_intersects_shape():
    mls = MultiGeoLineString(
        [
            GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5), Coordinate(1., 0.)]),
            GeoLineString([Coordinate(1., 1.), Coordinate(0.5, 0.5), Coordinate(0., 0.)]),
        ],
    )
    assert mls.intersects_shape(
        MultiGeoLineString(
            [
                GeoLineString([Coordinate(0., 0.5), Coordinate(1.0, 0.5)]),
            ],
        )
    )

    assert mls.intersects_shape(GeoLineString([Coordinate(0., 0.5), Coordinate(1.0, 0.5)]))
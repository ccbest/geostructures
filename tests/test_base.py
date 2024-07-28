
from datetime import datetime, timedelta, timezone

import pytest

from geostructures import *
from geostructures.multistructures import *
from geostructures.time import TimeInterval


def test_baseshapeprotocol_end():
    point = GeoPoint(Coordinate('0.0', '0.0'), dt=TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 2)))
    assert point.end == datetime(2020, 1, 2, tzinfo=timezone.utc)

    with pytest.raises(ValueError):
        _ = GeoPoint(Coordinate('0.0', '0.0')).end


def test_baseshapeprotocol_properties():
    # Base Case
    point = GeoPoint(Coordinate('0.0', '0.0'))
    assert point.properties == {}

    point = GeoPoint(Coordinate('0.0', '0.0'), dt=TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 1)))
    assert point.properties == {
        "datetime_start": datetime(2020, 1, 1, tzinfo=timezone.utc),
        "datetime_end": datetime(2020, 1, 1, tzinfo=timezone.utc),
    }

    point = GeoPoint(Coordinate('0.0', '0.0'), properties={'test': 'prop'})
    assert point.properties == {'test': 'prop'}


def test_baseshapeprotocol_start():
    point = GeoPoint(Coordinate('0.0', '0.0'), dt=TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 2)))
    assert point.start == datetime(2020, 1, 1, tzinfo=timezone.utc)

    with pytest.raises(ValueError):
        _ = GeoPoint(Coordinate('0.0', '0.0')).start


def test_baseshapeprotocol_buffer_dt():
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


def test_baseshapeprotocol_contains():
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


def test_baseshapeprotocol_contains_time():
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


def test_baseshapeprotocol_intersects_time():
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


def test_baseshapeprotocol_set_dt():
    point = GeoPoint(Coordinate('0.0', '0.0'))
    point2 = point.set_dt(datetime(2020, 1, 1))
    assert point2 is not point
    assert point2.dt == TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 1))
    assert point.dt is None

    point.set_dt(datetime(2020, 1, 1), inplace=True)
    assert point.dt == TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 1))

    with pytest.raises(ValueError):
        point.set_dt('not a date')


def test_baseshapeprotocol_set_property():
    point = GeoPoint(Coordinate('0.0', '0.0'), dt=datetime(2020, 1, 1, 12))
    assert point.properties == {
        'datetime_start': datetime(2020, 1, 1, 12, tzinfo=timezone.utc),
        'datetime_end': datetime(2020, 1, 1, 12, tzinfo=timezone.utc)
    }

    point.set_property('test_property', 1, inplace=True)
    assert point.properties == {
        'datetime_start': datetime(2020, 1, 1, 12, tzinfo=timezone.utc),
        'datetime_end': datetime(2020, 1, 1, 12, tzinfo=timezone.utc),
        'test_property': 1
    }


def test_baseshapeprotocol_strip_dt():
    point = GeoPoint(Coordinate('0.0', '0.0'), dt=datetime(2020, 1, 1, 12))
    expected = GeoPoint(Coordinate('0.0', '0.0'))
    assert point.strip_dt() == expected



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


def test_multigeobase_split():
    mps = MultiGeoPoint(
        [
            GeoPoint(Coordinate(0., 0.)),
            GeoPoint(Coordinate(1., 1.)),
        ],
        dt=TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 2)),
        properties={
            'test': 'prop'
        }
    )
    actual = mps.split()
    assert actual == [
        GeoPoint(
            Coordinate(0., 0.),
            dt=TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 2)),
            properties={
                'test': 'prop'
            }
        ),
        GeoPoint(
            Coordinate(1., 1.),
            dt=TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 2)),
            properties={
                'test': 'prop'
            }
        ),
    ]
    actual[0].set_property('test', 'diff', inplace=True)
    assert actual[0].properties['test'] != mps.properties['test']


from datetime import datetime

from geostructures import (
    Coordinate, GeoCircle, GeoLineString, GeoPoint,
    MultiGeoLineString, MultiGeoPoint,
)
from geostructures.time import TimeInterval


def test_multishapebase_contains():
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


def test_multishapebase_hash():
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


def test_multishapebase_bounds():
    mls = MultiGeoLineString(
        [
            GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5), Coordinate(1., 0.)]),
            GeoLineString([Coordinate(2., 1.), Coordinate(0.5, 0.5), Coordinate(0., 0.)]),
        ],
    )
    assert mls.bounds == (0., 0., 2., 1.)


def test_multishapebase_contains_coordinate():
    mls = MultiGeoLineString(
        [
            GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5), Coordinate(1., 0.)]),
            GeoLineString([Coordinate(1., 1.), Coordinate(0.5, 0.5), Coordinate(0., 0.)]),
        ],
    )
    assert mls.contains_coordinate(Coordinate(0.5, 0.5))
    assert not mls.contains_coordinate(Coordinate(2.0, 2.0))


def test_multishapebase_contains_shape():
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


def test_multishapebase_copy():
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


def test_multishapebase_intersects_shape():
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


def test_multishapebase_split():
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


def test_multishapebase_eq_non_multishape():
    mp = MultiGeoPoint([GeoPoint(Coordinate(0., 0.))])
    assert mp != 'not a multishape'


def test_multishapebase_contains_shape_negative():
    mls = MultiGeoLineString(
        [
            GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5), Coordinate(1., 0.)]),
        ],
    )
    # Multishape argument where not every subshape is contained
    assert not mls.contains_shape(
        MultiGeoPoint([GeoPoint(Coordinate(0., 1.)), GeoPoint(Coordinate(5., 5.))])
    )


def test_multishapebase_intersects_shape_negative():
    mls = MultiGeoLineString(
        [
            GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5), Coordinate(1., 0.)]),
        ],
    )
    assert not mls.intersects_shape(
        MultiGeoLineString([GeoLineString([Coordinate(5., 5.), Coordinate(6., 6.)])])
    )
    assert not mls.intersects_shape(GeoLineString([Coordinate(5., 5.), Coordinate(6., 6.)]))

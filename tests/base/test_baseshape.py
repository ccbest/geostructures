
from datetime import datetime, timedelta, timezone
import pickle
import tempfile

from fastkml import Placemark
from fastkml.times import TimeStamp, KmlDateTime
from fastkml.data import ExtendedData, Data
from pygeoif import MultiPolygon
import pytest
import shapely

from geostructures import Coordinate, GeoBox, GeoCircle, GeoPoint, MultiGeoPolygon
from geostructures.time import TimeInterval

from tests.functions import default_test_datetime


def test_baseshape_start():
    point = GeoPoint(Coordinate('0.0', '0.0'), dt=default_test_datetime)
    assert point.start == default_test_datetime

    point = GeoPoint(
        Coordinate('0.0', '0.0'),
        dt=TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 2))
    )
    assert point.start == datetime(2020, 1, 1, tzinfo=timezone.utc)

    with pytest.raises(ValueError):
        _ = GeoPoint(Coordinate('0.0', '0.0')).start


def test_baseshape_end():
    point = GeoPoint(Coordinate('0.0', '0.0'), dt=default_test_datetime)
    assert point.end == default_test_datetime

    point = GeoPoint(
        Coordinate('0.0', '0.0'),
        dt=TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 2))
    )
    assert point.end == datetime(2020, 1, 2, tzinfo=timezone.utc)

    with pytest.raises(ValueError):
        _ = GeoPoint(Coordinate('0.0', '0.0')).end


def test_baseshape_properties():
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


def test_baseshape_set_property():
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


def test_baseshape_buffer_dt():
    # Base Case
    point = GeoPoint(Coordinate('0.0', '0.0'), dt=TimeInterval(datetime(2020, 1, 1, 12), datetime(2020, 1, 3, 12)))
    point2 = point.buffer_dt(timedelta(hours=1), inplace=False)
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


def test_baseshape_set_dt():
    point = GeoPoint(Coordinate('0.0', '0.0'))
    point2 = point.set_dt(datetime(2020, 1, 1), inplace=False)
    assert point2 is not point
    assert point2.dt == TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 1))
    assert point.dt is None

    point.set_dt(datetime(2020, 1, 1), inplace=True)
    assert point.dt == TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 1))

    with pytest.raises(ValueError):
        point.set_dt('not a date')


def test_baseshape_strip_dt():
    point = GeoPoint(Coordinate('0.0', '0.0'), dt=datetime(2020, 1, 1, 12))
    expected = GeoPoint(Coordinate('0.0', '0.0'))
    assert point.strip_dt() == expected


def test_baseshape_contains():
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


def test_baseshape_contains_dunder():
    circle = GeoCircle(Coordinate('0.0', '0.0'), 500, dt=datetime(2020, 1, 1, 1))
    assert Coordinate('0.0', '0.0') in circle
    assert GeoPoint(Coordinate('0.0', '0.0'), dt=datetime(2020, 1, 1, 1)) in circle
    assert GeoPoint(Coordinate('0.0', '0.0'), dt=None) in circle
    assert GeoPoint(
        Coordinate('0.0', '0.0'), dt=datetime(2020, 1, 1, 1)
    ) in GeoCircle(Coordinate('0.0', '0.0'), 500, dt=None)


def test_baseshape_contains_time():
    geopoint = GeoPoint(Coordinate('0.0', '0.0'), dt=datetime(2020, 1, 1, 1))
    assert geopoint.contains_time(datetime(2020, 1, 1, 1))
    assert not geopoint.contains_time(datetime(2020, 1, 1, 1, 1))
    assert not geopoint.contains_time(TimeInterval(datetime(2020, 1, 1, 1), datetime(2020, 1, 1, 1, 1)))

    geopoint = GeoPoint(Coordinate('0.0', '0.0'), dt=TimeInterval(datetime(2020, 1, 1, 12), datetime(2020, 1, 3, 12)))
    assert geopoint.contains_time(datetime(2020, 1, 2))
    assert not geopoint.contains_time(datetime(2020, 1, 4, 12))
    assert geopoint.contains_time(TimeInterval(datetime(2020, 1, 1, 14), datetime(2020, 1, 1, 16)))
    assert not geopoint.contains_time(TimeInterval(datetime(2020, 1, 3, 11), datetime(2020, 1, 3, 14)))

    geopoint = GeoPoint(Coordinate('0.0', '0.0'), dt=None)
    assert not geopoint.contains_time(datetime(2020, 1, 4, 12))
    assert not geopoint.contains_time(TimeInterval(datetime(2020, 1, 1, 14), datetime(2020, 1, 1, 16)))

    with pytest.raises(ValueError):
        geopoint = GeoPoint(
            Coordinate('0.0', '0.0'),
            dt=TimeInterval(datetime(2020, 1, 1, 12), datetime(2020, 1, 3, 12))
        )
        geopoint.contains_time('not a date')


def test_baseshape_intersects_time():
    geopoint = GeoPoint(Coordinate('0.0', '0.0'), dt=datetime(2020, 1, 1, 1))
    assert geopoint.intersects_time(datetime(2020, 1, 1, 1))
    assert not geopoint.intersects_time(datetime(2020, 1, 1, 1, 1))
    assert geopoint.intersects_time(TimeInterval(datetime(2020, 1, 1, 1), datetime(2020, 1, 1, 1, 1)))

    geopoint = GeoPoint(Coordinate('0.0', '0.0'), dt=TimeInterval(datetime(2020, 1, 1, 12), datetime(2020, 1, 3, 12)))
    assert geopoint.intersects_time(datetime(2020, 1, 2))
    assert not geopoint.intersects_time(datetime(2020, 1, 4, 12))
    assert geopoint.intersects_time(TimeInterval(datetime(2020, 1, 1, 14), datetime(2020, 1, 1, 16)))

    geopoint = GeoPoint(Coordinate('0.0', '0.0'), dt=None)
    assert not geopoint.intersects_time(datetime(2020, 1, 4, 12))
    assert not geopoint.intersects_time(TimeInterval(datetime(2020, 1, 1, 14), datetime(2020, 1, 1, 16)))


def test_baseshape_intersects():
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


def test_baseshape_pickle():
    point = GeoPoint(
        Coordinate('0.0', '0.0'),
        dt=datetime(2020, 1, 1, 12),
        properties={'test': 'prop'}
    )
    with tempfile.TemporaryDirectory() as tempdir:
        with open(f'{tempdir}/temp.pkl', 'wb') as f:
            pickle.dump(point, f)

        with open(f'{tempdir}/temp.pkl', 'rb') as f:
            new_point = pickle.load(f)

        assert point == new_point
        assert point.properties == new_point.properties

        # Make sure the original wasn't mutated and the new call works
        assert point.to_shapely() == shapely.Point((0.0, 0.0))
        assert new_point.to_shapely() == shapely.Point((0.0, 0.0))


def test_baseshape_to_fastkml_placemark():
    shape = MultiGeoPolygon(
        [
            GeoBox(Coordinate(0., 1.), Coordinate(1., 0.)).to_polygon()
        ],
        dt=datetime(2020, 1, 1),
        properties={'test': 'prop'}
    )
    actual = shape.to_fastkml_placemark()
    expected = Placemark(
        geometry=MultiPolygon(
            [
                [
                    [x.to_float() for x in ring]
                    for ring in shape
                ] for shape in shape.linear_rings()
            ]
        ),
        times=TimeStamp(timestamp=KmlDateTime(datetime(2020, 1, 1, tzinfo=timezone.utc))),
        extended_data=ExtendedData(elements=[Data(name='test', value='prop')])
    )
    assert actual.geometry == expected.geometry
    assert actual.times == expected.times
    assert actual.extended_data == expected.extended_data
    assert actual == expected


from datetime import date, datetime, timedelta, timezone

from fastkml.times import TimeSpan, TimeStamp, KmlDateTime
import pytest

from geostructures.time import TimeInterval


def test_timeinterval_construction():
    interval = TimeInterval(datetime(2019, 1, 1), datetime(2019, 12, 31))
    assert interval == TimeInterval(datetime(2019, 1, 1), datetime(2019, 12, 31))

    interval = TimeInterval(datetime(2019, 1, 1), timedelta(days=2))
    assert interval == TimeInterval(datetime(2019, 1, 1), datetime(2019, 1, 3))

    with pytest.raises(ValueError):
        TimeInterval(datetime(2019, 12, 31), datetime(2019, 1, 1))


def test_timeinterval_eq():
    interval = TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 3))
    assert interval == TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 3))

    assert interval != TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 4))
    assert interval != TimeInterval(datetime(2020, 1, 2), datetime(2020, 1, 3))
    assert interval != 'not a timeinterval'


def test_timeeinterval_repr():
    t = TimeInterval(datetime(2023, 1, 2, 3, 4, 5, 6), datetime(2023, 1, 4, 3, 4, 5, 6))
    assert repr(t) == '<TimeInterval [2023-01-02T03:04:05.000006+00:00 - 2023-01-04T03:04:05.000006+00:00)>'


def test_timeinterval_contains():
    assert datetime(2020, 1, 2) in TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 3))
    assert datetime(2020, 1, 1) in TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 3))
    assert datetime(2020, 1, 3) not in TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 3))

    assert TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 3)) in TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 3))
    assert TimeInterval(datetime(2020, 1, 2), datetime(2020, 1, 2, 12)) in TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 3))
    assert TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 4)) not in TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 3))

    with pytest.raises(ValueError):
        _ = date(2020, 1, 1) in TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 3))

def test_timeinterval_hash():

    intervals = {
        TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 3)),
        TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 3)), # duplicate
        TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 4)),
        TimeInterval(datetime(2020, 1, 2), datetime(2020, 1, 3))
    }
    assert len(intervals) == 3
    assert TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 3)) in intervals
    assert TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 4)) in intervals
    assert TimeInterval(datetime(2020, 1, 2), datetime(2020, 1, 3)) in intervals


def test_timeinterval_copy():
    interval = TimeInterval(datetime(2020, 1, 2), datetime(2020, 1, 3))
    interval_copy = interval.copy()
    assert interval == interval_copy
    assert interval is not interval_copy


def test_timeinterval_isdisjoint():
    interval = TimeInterval(datetime(2020, 1, 2), datetime(2020, 1, 3))
    assert interval.isdisjoint(TimeInterval(datetime(2020, 1, 3), datetime(2020, 1, 4)))
    assert interval.isdisjoint(TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 2)))
    assert not interval.isdisjoint(TimeInterval(datetime(2020, 1, 2, 1), datetime(2020, 1, 2, 2)))


def test_timeinterval_issubset():
    interval = TimeInterval(datetime(2020, 1, 2), datetime(2020, 1, 3))
    assert interval.issubset(TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 4)))
    assert not interval.issubset(TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 2)))
    assert not interval.issubset(TimeInterval(datetime(2020, 1, 3), datetime(2020, 1, 4)))
    assert not interval.issubset(TimeInterval(datetime(2020, 1, 2, 1), datetime(2020, 1, 2, 2)))


def test_timeinterval_issuperset():
    interval = TimeInterval(datetime(2020, 1, 2), datetime(2020, 1, 3))
    assert interval.issuperset(TimeInterval(datetime(2020, 1, 2, 1), datetime(2020, 1, 2, 2)))
    assert not interval.issuperset(TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 4)))
    assert not interval.issuperset(TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 2)))
    assert not interval.issuperset(TimeInterval(datetime(2020, 1, 3), datetime(2020, 1, 4)))


def test_timeinterval_elapsed():
    assert TimeInterval(datetime(2020, 1, 2), datetime(2020, 1, 3)).elapsed == timedelta(days=1)


def test_timeinterval_union():
    interval = TimeInterval(datetime(2020, 1, 2), datetime(2020, 1, 3))
    assert interval.union(TimeInterval(datetime(2020, 1, 2), datetime(2020, 1, 3))) == interval
    assert interval.union(TimeInterval(datetime(2020, 1, 2, 1), datetime(2020, 1, 2, 2))) == interval
    assert interval.union(TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 4))) == TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 4))


def test_timeinterval_intersection():
    interval = TimeInterval(datetime(2020, 1, 2), datetime(2020, 1, 3))

    assert interval.intersection(TimeInterval(datetime(2020, 1, 2), datetime(2020, 1, 3))) == TimeInterval(datetime(2020, 1, 2), datetime(2020, 1, 3))
    assert interval.intersection(TimeInterval(datetime(2020, 1, 2, 5), datetime(2020, 1, 3))) == TimeInterval(datetime(2020, 1, 2, 5), datetime(2020, 1, 3))
    assert interval.intersection(TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 2, 1))) == TimeInterval(datetime(2020, 1, 2), datetime(2020, 1, 2, 1))
    assert interval.intersection(TimeInterval(datetime(2020, 1, 5), datetime(2020, 1, 6))) is None


def test_timeinterval_to_fastkml():
    interval = TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 1))
    expected = TimeStamp(timestamp=KmlDateTime(datetime(2020, 1, 1, tzinfo=timezone.utc)))
    assert interval._to_fastkml() == expected

    interval = TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 2))
    expected = TimeSpan(
        begin=KmlDateTime(datetime(2020, 1, 1, tzinfo=timezone.utc)),
        end=KmlDateTime(datetime(2020, 1, 2, tzinfo=timezone.utc))
    )
    assert interval._to_fastkml() == expected


def test_timeinterval_from_fastkml():
    timestamp = TimeStamp(timestamp=KmlDateTime(datetime(2020, 1, 1, tzinfo=timezone.utc)))
    expected = TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 1))
    assert TimeInterval._from_fastkml(timestamp) == expected

    timespan = TimeSpan(
        begin=KmlDateTime(datetime(2020, 1, 1, tzinfo=timezone.utc)),
        end=KmlDateTime(datetime(2020, 1, 2, tzinfo=timezone.utc))
    )
    expected = TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 2))
    assert TimeInterval._from_fastkml(timespan) == expected

    with pytest.raises(ValueError):
        TimeInterval._from_fastkml('something else')

def test_timeinterval_from_str():
    start = '2020-01-01T00:00:00.000'
    assert TimeInterval.from_str(start) == TimeInterval(
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        datetime(2020, 1, 1, tzinfo=timezone.utc)
    )

    # Test end + different format
    end = '2020-01-02 00:00:00.000'
    assert TimeInterval.from_str(start, end) == TimeInterval(
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        datetime(2020, 1, 2, tzinfo=timezone.utc)
    )

    # Custom format
    start = '2020/01/01 00:00:00.000'
    assert TimeInterval.from_str(start, time_format='%Y/%m/%d %H:%M:%S.%f') == TimeInterval(
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        datetime(2020, 1, 1, tzinfo=timezone.utc)
    )

    # Multiple custom formats
    end = '2020.01.01 00:00:00.000'
    expected = TimeInterval.from_str(start, end, time_format=['%Y/%m/%d %H:%M:%S.%f', '%Y.%m.%d %H:%M:%S.%f'])
    assert expected == TimeInterval(
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        datetime(2020, 1, 1, tzinfo=timezone.utc)
    )

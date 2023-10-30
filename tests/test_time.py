
import pytest
from datetime import date, datetime, timedelta
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
    assert TimeInterval(datetime(2020, 1, 2), datetime(2020, 1, 3)).elapsed() == timedelta(days=1)


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

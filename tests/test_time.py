
import pytest
from datetime import date, datetime, timedelta
from geostructures.time import DateInterval, TimeInterval


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


def test_dateinterval_init():
    dateinterval = DateInterval()
    assert dateinterval.start == date.min
    assert dateinterval.end == date.max

    dateinterval = DateInterval(end=date(2022, 1, 1))
    assert dateinterval.start == date.min
    assert dateinterval.end == date(2022, 1, 1)

    dateinterval = DateInterval(start=date(2022, 1, 1))
    assert dateinterval.start == date(2022, 1, 1)
    assert dateinterval.end == date.max

    # Assert works with timedeltas
    dateinterval = DateInterval(
        start=date(2022, 1, 1),
        end=timedelta(days=2)
    )
    assert dateinterval.start == date(2022, 1, 1)
    assert dateinterval.end == date(2022, 1, 3)

    # Make sure rounding to the nearest day works
    dateinterval = DateInterval(
        start=date(2022, 1, 1),
        end=timedelta(days=1, microseconds=1)
    )
    assert dateinterval.start == date(2022, 1, 1)
    assert dateinterval.end == date(2022, 1, 2)

    # exactly noon, combo of seconds and microseconds - round up
    dateinterval = DateInterval(
        start=date(2022, 1, 1),
        end=timedelta(days=1, seconds=43199, microseconds=1_000_000)
    )
    assert dateinterval.start == date(2022, 1, 1)
    assert dateinterval.end == date(2022, 1, 2)

    # > noon
    dateinterval = DateInterval(
        start=date(2022, 1, 1),
        end=timedelta(days=1, seconds=44000, microseconds=1)
    )
    assert dateinterval.start == date(2022, 1, 1)
    assert dateinterval.end == date(2022, 1, 2)

    dateinterval = DateInterval(
        start=date(2022, 1, 1),
        end=2
    )
    assert dateinterval.start == date(2022, 1, 1)
    assert dateinterval.end == date(2022, 1, 3)

    with pytest.raises(ValueError):
        _ = DateInterval(date(2022, 1, 3), date(2022, 1, 1))


def test_dateinterval_hash():
    set1 = {
        DateInterval(date(2022, 1, 1), date(2022, 1, 2)),
        DateInterval(date(2022, 1, 1), date(2022, 1, 2)),  # duplicate
        DateInterval(date(2022, 1, 2), date(2022, 1, 3)),
        DateInterval(date(2022, 1, 3), date(2022, 1, 4)),
    }
    assert len(set1) == 3
    assert DateInterval(date(2022, 1, 1), date(2022, 1, 2)) in set1
    assert DateInterval(date(2022, 1, 2), date(2022, 1, 3)) in set1
    assert DateInterval(date(2022, 1, 3), date(2022, 1, 4)) in set1


def test_dateinterval_contains():
    dateinterval = DateInterval(
        start=date(2022, 1, 1),
        end=date(2022, 1, 3)
    )
    assert date(2022, 1, 1) in dateinterval
    assert date(2022, 1, 2) in dateinterval
    assert date(2022, 1, 3) in dateinterval
    assert date(2022, 1, 4) not in dateinterval


def test_dateinterval_eq():
    dateinterval = DateInterval(
        start=date(2022, 1, 1),
        end=date(2022, 1, 3)
    )
    dateinterval2 = DateInterval(
        start=date(2022, 1, 1),
        end=date(2022, 1, 3)
    )
    assert dateinterval == dateinterval2

    assert dateinterval != 5


def test_dateinterval_repr():
    dateinterval = DateInterval(
        start=date(2022, 1, 1),
        end=date(2022, 1, 3)
    )
    assert repr(dateinterval) == '<DateInterval [2022-01-01 - 2022-01-03]>'


def test_dateinterval_disjointed():
    dateinterval_1 = DateInterval(
        start=date(2022, 1, 1),
        end=date(2022, 1, 3)
    )
    dateinterval_2 = DateInterval(
        start=date(2022, 1, 1),
        end=date(2022, 1, 2)
    )
    assert not dateinterval_1.isdisjoint(dateinterval_2)

    dateinterval_1 = DateInterval(
        start=date(2022, 1, 1),
        end=date(2022, 1, 3)
    )
    dateinterval_2 = DateInterval(
        start=date(2022, 1, 6),
        end=date(2022, 1, 7)
    )
    assert dateinterval_1.isdisjoint(dateinterval_2)


def test_dateinterval_issubset():
    dateinterval_1 = DateInterval(
        start=date(2022, 1, 1),
        end=date(2022, 1, 5)
    )
    dateinterval_2 = DateInterval(
        start=date(2022, 1, 2),
        end=date(2022, 1, 3)
    )
    assert dateinterval_2.issubset(dateinterval_1)

    # end does not overlap
    dateinterval_1 = DateInterval(
        start=date(2022, 1, 1),
        end=date(2022, 1, 2)
    )
    dateinterval_2 = DateInterval(
        start=date(2022, 1, 2),
        end=date(2022, 1, 3)
    )
    assert not dateinterval_2.issubset(dateinterval_1)

    # start does not overlap
    dateinterval_1 = DateInterval(
        start=date(2022, 1, 2),
        end=date(2022, 1, 3)
    )
    dateinterval_2 = DateInterval(
        start=date(2022, 1, 1),
        end=date(2022, 1, 2)
    )
    assert not dateinterval_2.issubset(dateinterval_1)


def test_dateinterval_issuperset():
    dateinterval_1 = DateInterval(
        start=date(2022, 1, 1),
        end=date(2022, 1, 5)
    )
    dateinterval_2 = DateInterval(
        start=date(2022, 1, 2),
        end=date(2022, 1, 3)
    )
    assert dateinterval_1.issuperset(dateinterval_2)

    # end does not overlap
    dateinterval_1 = DateInterval(
        start=date(2022, 1, 1),
        end=date(2022, 1, 3),
    )
    dateinterval_2 = DateInterval(
        start=date(2022, 1, 2),
        end=date(2022, 1, 4)
    )
    assert not dateinterval_1.issuperset(dateinterval_2)

    # start does not overlap
    dateinterval_1 = DateInterval(
        start=date(2022, 1, 2),
        end=date(2022, 1, 3)
    )
    dateinterval_2 = DateInterval(
        start=date(2022, 1, 1),
        end=date(2022, 1, 2)
    )
    assert not dateinterval_1.issuperset(dateinterval_2)


def test_dateinterval_elapsed():
    dateinterval = DateInterval(
        start=date(2022, 1, 2),
        end=date(2022, 1, 3)
    )
    assert dateinterval.elapsed == timedelta(days=2)

    dateinterval = DateInterval(
        start=date(2022, 1, 2),
        end=date(2022, 1, 2),
    )
    assert dateinterval.elapsed == timedelta(days=1)


def test_dateinterval_union():
    # self.end > interval.end
    dateinterval_1 = DateInterval(
        start=date(2022, 1, 2),
        end=date(2022, 1, 4)
    )
    dateinterval_2 = DateInterval(
        start=date(2022, 1, 1),
        end=date(2022, 1, 3)
    )
    assert dateinterval_1.union(dateinterval_2) == DateInterval(
        start=date(2022, 1, 1),
        end=date(2022, 1, 4)
    )

    # self.end < interval.end
    dateinterval_1 = DateInterval(
        start=date(2022, 1, 1),
        end=date(2022, 1, 3)
    )
    dateinterval_2 = DateInterval(
        start=date(2022, 1, 1),
        end=date(2022, 1, 5)
    )
    assert dateinterval_1.union(dateinterval_2) == DateInterval(
        start=date(2022, 1, 1),
        end=date(2022, 1, 5)
    )
    assert dateinterval_2.union(dateinterval_1) == DateInterval(
        start=date(2022, 1, 1),
        end=date(2022, 1, 5)
    )


def test_dateinterval_intersection():
    dateinterval_1 = DateInterval(
        start=date(2022, 1, 2),
        end=date(2022, 1, 3)
    )
    dateinterval_2 = DateInterval(
        start=date(2022, 1, 1),
        end=date(2022, 1, 3)
    )
    assert dateinterval_1.intersection(dateinterval_2) == DateInterval(
        start=date(2022, 1, 2),
        end=date(2022, 1, 3)
    )

    dateinterval_1 = DateInterval(
        start=date(2022, 1, 2),
        end=date(2022, 1, 3)
    )
    dateinterval_2 = DateInterval(
        start=date(2022, 1, 6),
        end=date(2022, 1, 7),
    )
    assert dateinterval_1.intersection(dateinterval_2) is None

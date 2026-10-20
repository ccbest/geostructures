
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


def test_timeinterval_from_fastkml_open_ended():
    # KML allows open-ended TimeSpans; the known endpoint becomes an instant
    begin_only = TimeSpan(begin=KmlDateTime(datetime(2020, 1, 1, tzinfo=timezone.utc)))
    expected = TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 1))
    assert TimeInterval._from_fastkml(begin_only) == expected

    end_only = TimeSpan(end=KmlDateTime(datetime(2020, 1, 2, tzinfo=timezone.utc)))
    expected = TimeInterval(datetime(2020, 1, 2), datetime(2020, 1, 2))
    assert TimeInterval._from_fastkml(end_only) == expected

    # Time objects carrying no time information at all produce no interval
    assert TimeInterval._from_fastkml(TimeSpan()) is None
    assert TimeInterval._from_fastkml(TimeStamp()) is None


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

    with pytest.raises(ValueError):
        # Bad format passed
        TimeInterval.from_str(start, time_format='2020.01.01 00:00:00.000')


def test_timeinterval_aware_naive_mix():
    # Aware datetimes used to raise a TypeError when compared against the
    # naive datetime.min/datetime.max defaults before normalization
    interval = TimeInterval(datetime(2020, 1, 1, tzinfo=timezone.utc))
    assert interval.start == datetime(2020, 1, 1, tzinfo=timezone.utc)
    assert interval.end.tzinfo is not None

    interval = TimeInterval(end=datetime(2020, 1, 1, tzinfo=timezone.utc))
    assert interval.start.tzinfo is not None

    # Mixed aware/naive arguments normalize to UTC before comparison
    interval = TimeInterval(
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        datetime(2020, 1, 2)
    )
    assert interval.end == datetime(2020, 1, 2, tzinfo=timezone.utc)

    with pytest.raises(ValueError):
        TimeInterval(
            datetime(2020, 1, 2, tzinfo=timezone.utc),
            datetime(2020, 1, 1)
        )


def test_timeinterval_boundary_semantics():
    # Right-open: adjacent intervals are disjoint and their intersection is None
    a = TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 2))
    b = TimeInterval(datetime(2020, 1, 2), datetime(2020, 1, 3))
    assert a.isdisjoint(b)
    assert a.intersection(b) is None
    assert not a.intersects(b)

    # An instant on the exclusive end bound is disjoint; on the inclusive
    # start bound it intersects
    instant = TimeInterval(datetime(2020, 1, 2), datetime(2020, 1, 2))
    assert a.isdisjoint(instant)
    assert not a.intersects(instant)
    assert b.intersects(instant)

    # Instants intersect only when equal
    assert instant.intersects(TimeInterval(datetime(2020, 1, 2), datetime(2020, 1, 2)))
    assert instant.isdisjoint(TimeInterval(datetime(2020, 1, 2, 1), datetime(2020, 1, 2, 1)))


def test_timeinterval_from_str_plain_iso():
    expected = TimeInterval(datetime(2020, 1, 1, 12), datetime(2020, 1, 1, 12))
    assert TimeInterval.from_str('2020-01-01T12:00:00') == expected
    assert TimeInterval.from_str('2020-01-01 12:00:00') == expected


def test_parse_timestamp_does_not_mutate_formats():
    from geostructures.time import _DEFAULT_DATE_FORMATS

    before = list(_DEFAULT_DATE_FORMATS)
    TimeInterval._parse_timestamp('2020-01-01')
    assert _DEFAULT_DATE_FORMATS == before

    custom = ['%Y-%m-%dT%H:%M:%S', '%Y']
    TimeInterval._parse_timestamp('2020', custom)
    assert custom == ['%Y-%m-%dT%H:%M:%S', '%Y']


def test_timeinterval_preserves_timezones():
    from zoneinfo import ZoneInfo

    eastern = ZoneInfo('America/New_York')
    start = datetime(2020, 3, 8, 1, 30, tzinfo=eastern)   # before the DST change
    end = datetime(2020, 3, 8, 12, tzinfo=timezone(timedelta(hours=5)))
    interval = TimeInterval(start, end)

    assert interval.start == start and interval.start.tzinfo is eastern
    assert interval.start.utcoffset() == timedelta(hours=-5)
    assert interval.end == end and interval.end.utcoffset() == timedelta(hours=5)

    # Naive datetimes are treated as UTC
    assert TimeInterval(datetime(2020, 1, 1)).start == datetime(2020, 1, 1, tzinfo=timezone.utc)

    # Equality is by instant, regardless of the timezone used to express it
    assert TimeInterval(start, start) == TimeInterval(
        start.astimezone(timezone.utc), start.astimezone(timezone.utc)
    )
    assert hash(TimeInterval(start, start)) == hash(
        TimeInterval(start.astimezone(timezone.utc), start.astimezone(timezone.utc))
    )

    # A timedelta end is added to the instant, in the start's timezone, so the
    # interval lasts exactly that long across a DST change
    interval = TimeInterval(start, timedelta(hours=1))
    assert interval.elapsed == timedelta(hours=1)
    assert interval.end == datetime(2020, 3, 8, 3, 30, tzinfo=eastern)
    assert interval.end.tzinfo is eastern

    # Including from a repeated hour, where wall-clock arithmetic ends before the start
    repeated = datetime(2020, 11, 1, 1, 30, fold=1, tzinfo=eastern)
    assert TimeInterval(repeated, timedelta(minutes=10)).elapsed == timedelta(minutes=10)


def test_timeinterval_set_operations_keep_endpoint_timezones():
    plus_one = timezone(timedelta(hours=1))
    a = TimeInterval(datetime(2020, 1, 1, tzinfo=plus_one), datetime(2020, 1, 3, tzinfo=plus_one))
    b = TimeInterval(datetime(2020, 1, 2), datetime(2020, 1, 4))

    assert a.intersection(b).start.tzinfo is timezone.utc
    assert a.intersection(b).end.tzinfo is plus_one
    assert a.union(b).start.tzinfo is plus_one
    assert a.union(b).end.tzinfo is timezone.utc
    assert a.copy() == a and a.copy().start.tzinfo is plus_one
    assert a.elapsed == timedelta(days=2)


def test_timeinterval_is_read_only():
    interval = TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 2))
    with pytest.raises(AttributeError):
        interval.start = datetime(2019, 1, 1)
    with pytest.raises(AttributeError):
        interval.end = datetime(2021, 1, 1)


def test_timeinterval_pandas_timestamps():
    import pandas as pd

    start = pd.Timestamp('2020-01-01 00:00:00.000001999', tz='Europe/Paris')
    interval = TimeInterval(start, pd.Timestamp('2020-01-02'))

    # Returned as plain datetimes, in the original timezone, to the microsecond
    assert type(interval.start) is datetime
    assert interval.start == datetime(2019, 12, 31, 23, 0, 0, 1, tzinfo=timezone.utc)
    assert interval.start.utcoffset() == timedelta(hours=1)
    assert interval.end == datetime(2020, 1, 2, tzinfo=timezone.utc)

    with pytest.raises(ValueError):
        TimeInterval(pd.NaT, datetime(2020, 1, 1))

    # NaT isn't in any interval
    assert pd.NaT not in interval
    assert pd.NaT not in TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 1))

    # Timestamps outside the nanosecond range (1677-2262), and in other units
    old = pd.Timestamp(datetime(1500, 1, 1, 0, 0, 0, 5))
    assert TimeInterval(old, old).start == datetime(1500, 1, 1, 0, 0, 0, 5, tzinfo=timezone.utc)
    future = pd.Timestamp(datetime(3000, 1, 1)).as_unit('s')
    assert TimeInterval(future).start == datetime(3000, 1, 1, tzinfo=timezone.utc)

    # Sub-microsecond digits are floored before the epoch too
    before_epoch = pd.Timestamp('1969-12-31 23:59:59.9999995', tz='UTC')
    assert TimeInterval(before_epoch).start == datetime(1969, 12, 31, 23, 59, 59, 999999, tzinfo=timezone.utc)

    # A timedelta end is added to the instant, as pandas does, across a DST change
    fall_back = pd.Timestamp('2020-11-01 00:30', tz='America/New_York')
    interval = TimeInterval(fall_back, timedelta(hours=2))
    assert interval.elapsed == timedelta(hours=2)
    assert interval.end == fall_back + timedelta(hours=2)


def test_timeinterval_coerces_like_pydantic():
    assert TimeInterval('2020-01-01T00:00:00Z', '2020-01-02T00:00:00Z') == TimeInterval(
        datetime(2020, 1, 1), datetime(2020, 1, 2)
    )
    with pytest.raises(ValueError):
        TimeInterval('not a date')


def test_timeinterval_limits():
    interval = TimeInterval()
    assert interval.start == datetime.min.replace(tzinfo=timezone.utc)
    assert interval.end == datetime.max.replace(tzinfo=timezone.utc)

    # Instants that can't be expressed as a datetime in UTC are rejected up front
    with pytest.raises(ValueError):
        TimeInterval(datetime.min.replace(tzinfo=timezone(timedelta(hours=1))))
    with pytest.raises(ValueError):
        TimeInterval(end=datetime.max.replace(tzinfo=timezone(timedelta(hours=-1))))
    with pytest.raises(ValueError):
        TimeInterval(datetime(9999, 12, 31), timedelta(days=2))


def test_timeinterval_timedelta_end_error_message():
    with pytest.raises(ValueError, match=r'end date 2019-12-31 00:00:00\+00:00 must not be less than start date'):
        TimeInterval(datetime(2020, 1, 1), timedelta(days=-1))


def test_timeinterval_pickle():
    import pickle

    plus_one = timezone(timedelta(hours=1))
    interval = TimeInterval(datetime(2020, 1, 1, tzinfo=plus_one), datetime(2020, 1, 2))
    restored = pickle.loads(pickle.dumps(interval))
    assert restored == interval and restored.start.tzinfo == plus_one

    # Intervals pickled before the integer representation stored datetimes
    legacy = TimeInterval.__new__(TimeInterval)
    legacy.__setstate__({'start': interval.start, 'end': interval.end})
    assert legacy == interval and legacy.start.tzinfo == plus_one


def test_timeinterval_caches_endpoints():
    import pickle

    plus_one = timezone(timedelta(hours=1))
    interval = TimeInterval(datetime(2020, 1, 1, tzinfo=plus_one), datetime(2020, 1, 2))
    assert interval.start is interval.start and interval.end is interval.end

    # Cached datetimes aren't pickled, and are rebuilt on access
    assert len(interval.__getstate__()) == 4
    restored = pickle.loads(pickle.dumps(interval))
    assert restored.start == interval.start and restored.start.tzinfo == plus_one
    assert interval.copy().end == interval.end

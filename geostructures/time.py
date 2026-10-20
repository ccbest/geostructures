"""Data structures for time intervals and milestones"""

from __future__ import annotations

__all__ = ['TimeInterval', 'GEOTIME_TYPE']

from datetime import datetime, timedelta, timezone, tzinfo
from typing import List, Optional, Tuple, Union, cast

from pydantic import TypeAdapter
from geostructures.utils.conditional_imports import import_optional


_DEFAULT_DATE_FORMATS = [
    '%Y-%m-%dT%H:%M:%S.%f%z',
    '%Y-%m-%dT%H:%M:%S.%f',
    '%Y-%m-%dT%H:%M:%S%z',
    '%Y-%m-%dT%H:%M:%S',
    '%Y-%m-%d %H:%M:%S.%f%z',
    '%Y-%m-%d %H:%M:%S.%f',
    '%Y-%m-%d %H:%M:%S%z',
    '%Y-%m-%d %H:%M:%S',
    '%Y-%m-%d',
]

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_UTC = timezone.utc
# Converting to a timezone shifts by less than a day, so only instants within a
# day of datetime's limits can fail to convert back
_SAFE_MIN = (datetime.min.replace(tzinfo=_UTC) - _EPOCH) // timedelta(microseconds=1) + 86_400_000_000
_SAFE_MAX = (datetime.max.replace(tzinfo=_UTC) - _EPOCH) // timedelta(microseconds=1) - 86_400_000_000

# Lax coercion for values that aren't already datetimes/timedeltas (e.g. ISO
# strings), matching what the constructor accepted when validated by pydantic
_START_ADAPTER = TypeAdapter(datetime)
_END_ADAPTER = TypeAdapter(Union[datetime, timedelta])  # type: ignore


def _as_aware(dt: datetime) -> datetime:
    """Treat a naive datetime as UTC"""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=_UTC)


def _to_micros(dt: datetime) -> int:
    """
    Microseconds since the Unix epoch, for a timezone-aware datetime or
    pandas Timestamp. Digits below a microsecond are dropped (floored).
    """
    if type(dt) is not datetime:  # pylint: disable=unidiomatic-typecheck
        asm8 = getattr(dt, 'asm8', None)
        if asm8 is not None:
            # pandas Timestamp, as a numpy datetime64 (UTC) in its own unit;
            # converting the unit floors, and avoids .value, which overflows
            # outside the nanosecond range
            if dt != dt:  # pylint: disable=comparison-with-itself
                raise ValueError('TimeInterval endpoints may not be NaT')
            return int(asm8.astype('datetime64[us]').astype('int64'))

    delta = dt - _EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


def _to_datetime(micros: int, tz: tzinfo) -> datetime:
    """Inverse of _to_micros, in the given timezone"""
    return (_EPOCH + timedelta(microseconds=micros)).astimezone(tz)


def _representable(micros: int, tz: tzinfo) -> bool:
    """Whether an instant can be expressed as a datetime in the given timezone"""
    if _SAFE_MIN < micros < _SAFE_MAX:
        return True
    try:
        _to_datetime(micros, tz)
    except OverflowError:
        return False
    return True


def _endpoint(dt: datetime) -> Tuple[int, tzinfo]:
    """The (microseconds, timezone) a datetime is stored as; naive datetimes are treated as UTC"""
    dt = _as_aware(dt)
    tz = cast(tzinfo, dt.tzinfo)
    micros = _to_micros(dt)
    # Any datetime in UTC can be converted back, so only other timezones need checking
    if tz is not _UTC and not _representable(micros, tz):
        raise ValueError(f'{dt} lies outside the range of datetimes representable in UTC')
    return micros, tz


class TimeInterval:
    """
    A class representing a right-open time interval.

    Each endpoint is stored as an integer count of microseconds since the
    Unix epoch (UTC), alongside the timezone it was provided in. The start and
    end properties return datetimes in that original timezone; naive
    datetimes are treated as UTC. Each is rebuilt on first access and cached.
    """

    __slots__ = ('_start', '_end', '_start_tz', '_end_tz', '_start_dt', '_end_dt')

    _start: int
    _end: int
    _start_tz: tzinfo
    _end_tz: tzinfo
    _start_dt: Optional[datetime]
    _end_dt: Optional[datetime]

    def __init__(
        self,
        start: datetime = datetime.min,
        end: Union[datetime, timedelta] = datetime.max,
    ):
        """Check to make sure end is after start"""
        if not isinstance(start, datetime):
            start = _START_ADAPTER.validate_python(start)
        if not isinstance(end, (datetime, timedelta)):
            end = _END_ADAPTER.validate_python(end)

        start_micros, start_tz = _endpoint(start)
        if isinstance(end, timedelta):
            # Added to the instant, so the interval lasts exactly that long even
            # across a daylight saving change
            end_micros, end_tz = start_micros + end // timedelta(microseconds=1), start_tz
            if not _representable(end_micros, end_tz):
                raise ValueError(f'{start} + {end} lies outside the range of representable datetimes')
        else:
            end_micros, end_tz = _endpoint(end)

        if end_micros < start_micros:
            raise ValueError(
                f'end date {_to_datetime(end_micros, end_tz)} must not be less than '
                f'start date {_to_datetime(start_micros, start_tz)}'
            )

        self._start, self._start_tz = start_micros, start_tz
        self._end, self._end_tz = end_micros, end_tz
        self._start_dt = self._end_dt = None

    @classmethod
    def _from_micros(
        cls,
        start: int,
        end: int,
        start_tz: tzinfo = _UTC,
        end_tz: tzinfo = _UTC,
    ) -> TimeInterval:
        """
        Create an interval directly from its stored representation, skipping
        validation. Callers are responsible for end >= start and both
        endpoints being representable as datetimes.
        """
        new = cls.__new__(cls)
        new._start, new._end = start, end
        new._start_tz, new._end_tz = start_tz, end_tz
        new._start_dt = new._end_dt = None
        return new

    @property
    def start(self) -> datetime:
        """The (inclusive) start of the interval, in the timezone it was provided in"""
        dt = self._start_dt
        if dt is None:
            dt = self._start_dt = _to_datetime(self._start, self._start_tz)
        return dt

    @property
    def end(self) -> datetime:
        """The (exclusive) end of the interval, in the timezone it was provided in"""
        dt = self._end_dt
        if dt is None:
            dt = self._end_dt = _to_datetime(self._end, self._end_tz)
        return dt

    def __getstate__(self):
        return self._start, self._end, self._start_tz, self._end_tz

    def __setstate__(self, state):
        self._start_dt = self._end_dt = None
        if isinstance(state, dict):
            # Pickled before intervals stored microseconds: {'start': ..., 'end': ...}
            (self._start, self._start_tz), (self._end, self._end_tz) = (
                _endpoint(state['start']), _endpoint(state['end'])
            )
            return
        self._start, self._end, self._start_tz, self._end_tz = state

    def __eq__(self, other) -> bool:
        """Test equality"""
        if not isinstance(other, TimeInterval):
            return False

        return self._start == other._start and self._end == other._end

    def __repr__(self):
        """REPL representation"""
        return f'<TimeInterval [{self.start.isoformat()} - {self.end.isoformat()})>'

    def __contains__(self, time: Union[datetime, TimeInterval]) -> bool:
        """Returns true if datetime is within range"""
        if isinstance(time, datetime):
            if time != time:  # pylint: disable=comparison-with-itself
                # pandas NaT, which isn't in any interval
                return False
            micros = _to_micros(_as_aware(time))
            if self.is_instant:
                return self._start == micros
            return self._start <= micros < self._end

        if isinstance(time, TimeInterval):
            return self.issuperset(time)

        raise ValueError('TimeIntervals may only contain datetimes and other TimeIntervals.')

    def __hash__(self) -> int:
        return hash((self._start, self._end))

    @property
    def elapsed(self) -> timedelta:
        """Returns elapsed time (as a timedelta) for the interval"""
        return timedelta(microseconds=self._end - self._start)

    @property
    def is_instant(self) -> bool:
        return self._start == self._end

    @staticmethod
    def _from_fastkml(fastkml_time):
        """
        Create a TimeInterval from the FastKML equivalent, or None if the
        FastKML object carries no time information.

        KML allows open-ended TimeSpans (only a <begin> or only an <end>);
        TimeIntervals are always bounded, so these become zero-length
        intervals at the known endpoint.
        """
        import_optional('fastkml')
        from fastkml.times import TimeSpan, TimeStamp

        if isinstance(fastkml_time, TimeStamp):
            if fastkml_time.timestamp is None:
                return None
            return TimeInterval(fastkml_time.timestamp.dt, fastkml_time.timestamp.dt)

        if isinstance(fastkml_time, TimeSpan):
            begin = fastkml_time.begin.dt if fastkml_time.begin is not None else None
            end = fastkml_time.end.dt if fastkml_time.end is not None else None
            if begin is None and end is None:
                return None
            if begin is None:
                begin = end
            if end is None:
                end = begin
            return TimeInterval(begin, end)

        raise ValueError('Unrecognized FastKML time object.')

    @staticmethod
    def _parse_timestamp(
        time_str: str,
        formats: Optional[List[str]] = None
    ) -> datetime:
        formats = formats or _DEFAULT_DATE_FORMATS
        for fmt in formats:
            try:
                return datetime.strptime(time_str, fmt)
            except ValueError:
                continue
        raise ValueError(f'Date format was not recognized; {time_str}')

    def _to_fastkml(self):
        """Convert to a FastKML equivalent object."""
        import_optional('fastkml')
        from fastkml.times import TimeSpan, TimeStamp, KmlDateTime

        if self.is_instant:
            return TimeStamp(timestamp=KmlDateTime(dt=self.start))

        return TimeSpan(begin=KmlDateTime(self.start), end=KmlDateTime(self.end))

    def copy(self):
        return self._from_micros(self._start, self._end, self._start_tz, self._end_tz)

    @classmethod
    def from_str(
        cls,
        start: str,
        end: Optional[str] = None,
        time_format: Optional[Union[str, List[str]]] = None
    ) -> GEOTIME_TYPE:
        """
        Create a TimeInterval from stringified timestamp(s). A limited number
        of default common timestamp formats will be checked; you may pass one or
        more custom formats to use instead.

        Uses standard python strptime format codes, documented here:
        https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes

        Args:
            start: (str)
                The start time, as a string

            end: (str) (Optional)
                The end time, as a string. If no end time is provided, the start time be also
                be treated as the end time.

            time_format: (Union[str, List[str]]) (Optional)
                Any custom timestamp formats to attempt parsing with.

        Returns:
            TimeInterval
        """
        if isinstance(time_format, str):
            time_format = [time_format]

        parsed_start = cls._parse_timestamp(start, time_format)
        if end is None:
            return cls(parsed_start, parsed_start)

        parsed_end = cls._parse_timestamp(end, time_format)
        return cls(parsed_start, parsed_end)

    def intersects(self, other: Union[datetime, TimeInterval]) -> bool:
        """Test whether this time interval intersects with another interval or point in time"""
        if isinstance(other, datetime):
            return other in self
        return not other.isdisjoint(self)

    def intersection(self, other: TimeInterval) -> Optional[TimeInterval]:
        """Returns a TimeInterval common to both time intervals. Will return None if impossible"""
        if self.isdisjoint(other):
            return None
        # Ties keep self's endpoint (and so its timezone), as max()/min() would
        start = other if other._start > self._start else self
        end = other if other._end < self._end else self
        return self._from_micros(start._start, end._end, start._start_tz, end._end_tz)

    def isdisjoint(self, other: TimeInterval) -> bool:
        """
        Returns True if time intervals do not overlap.

        Intervals are right-open, so an instant sitting exactly on another
        interval's (exclusive) end bound is disjoint from it. Two instants
        are disjoint unless they are equal.
        """
        if self.is_instant and other.is_instant:
            return self._start != other._start
        if self.is_instant:
            return not other._start <= self._start < other._end
        if other.is_instant:
            return not self._start <= other._start < self._end
        return self._end <= other._start or self._start >= other._end

    def issubset(self, other: TimeInterval) -> bool:
        """Returns True if time interval is contained entirely within other interval"""
        return (other._start <= self._start) and (self._end <= other._end)

    def issuperset(self, other: TimeInterval) -> bool:
        """Returns True if other interval is contained entirely within time interval"""
        return other.issubset(self)

    def union(self, other: TimeInterval) -> TimeInterval:
        """Returns a TimeInterval that spans both time intervals"""
        start = other if other._start < self._start else self
        end = other if other._end > self._end else self
        return self._from_micros(start._start, end._end, start._start_tz, end._end_tz)


GEOTIME_TYPE = Union[datetime, TimeInterval]

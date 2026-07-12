"""Data structures for time intervals and milestones"""

from __future__ import annotations

__all__ = ['TimeInterval', 'GEOTIME_TYPE']

from datetime import datetime, timedelta, timezone
from typing import List, Optional, Union

from pydantic import validate_call
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


class TimeInterval:
    """A class representing a right-open time interval"""

    @validate_call
    def __init__(
        self,
        start: datetime = datetime.min,
        end: Union[datetime, timedelta] = datetime.max,
    ):
        """Check to make sure end is after start"""
        super().__init__()
        # Normalize to timezone-aware before comparing; a naive/aware mix
        # (e.g. an aware start with the naive datetime.max default) raises
        # an unhelpful TypeError otherwise
        start = self._default_to_zulu(start)
        end = self._default_to_zulu(end if isinstance(end, datetime) else start + end)

        if end < start:
            raise ValueError(f'end date {end} must not be less than start date {start}')

        self.start, self.end = start, end

    def _default_to_zulu(self, dt: datetime) -> datetime:
        """Add Zulu/UTC as timezone, if timezone not present"""
        if not dt.tzinfo:
            return dt.replace(tzinfo=timezone.utc)

        return dt

    def __eq__(self, other) -> bool:
        """Test equality"""
        if not isinstance(other, TimeInterval):
            return False

        return self.start == other.start and self.end == other.end

    def __repr__(self):
        """REPL representation"""
        return f'<TimeInterval [{self.start.isoformat()} - {self.end.isoformat()})>'

    def __contains__(self, time: Union[datetime, TimeInterval]) -> bool:
        """Returns true if datetime is within range"""
        if isinstance(time, datetime):
            time = self._default_to_zulu(time)
            if self.is_instant:
                return self.start == time
            return self.start <= time < self.end

        if isinstance(time, TimeInterval):
            return self.issuperset(time)

        raise ValueError('TimeIntervals may only contain datetimes and other TimeIntervals.')

    def __hash__(self) -> int:
        return hash((self.start, self.end))

    @property
    def elapsed(self):
        """Returns elapsed time (as a timedelta) for the interval"""
        return self.end - self.start

    @property
    def is_instant(self):
        return self.start == self.end

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

        if self.start == self.end:
            return TimeStamp(timestamp=KmlDateTime(dt=self.start))

        return TimeSpan(begin=KmlDateTime(self.start), end=KmlDateTime(self.end))

    def copy(self):
        return TimeInterval(self.start, self.end)

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
        return TimeInterval(max(self.start, other.start), min(self.end, other.end))

    def isdisjoint(self, other: TimeInterval) -> bool:
        """
        Returns True if time intervals do not overlap.

        Intervals are right-open, so an instant sitting exactly on another
        interval's (exclusive) end bound is disjoint from it. Two instants
        are disjoint unless they are equal.
        """
        if self.is_instant and other.is_instant:
            return self.start != other.start
        if self.is_instant:
            return self.start not in other
        if other.is_instant:
            return other.start not in self
        return self.end <= other.start or self.start >= other.end

    def issubset(self, other: TimeInterval) -> bool:
        """Returns True if time interval is contained entirely within other interval"""
        return (other.start <= self.start) and (self.end <= other.end)

    def issuperset(self, other: TimeInterval) -> bool:
        """Returns True if other interval is contained entirely within time interval"""
        return other.issubset(self)

    def union(self, other: TimeInterval) -> TimeInterval:
        """Returns a TimeInterval that spans both time intervals"""
        return TimeInterval(min(self.start, other.start), max(self.end, other.end))


GEOTIME_TYPE = Union[datetime, TimeInterval]

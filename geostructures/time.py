"""Data structures for time intervals and milestones"""

from __future__ import annotations

__all__ = ['TimeInterval', 'GEOTIME_TYPE']

from datetime import datetime, timedelta, timezone
from typing import List, Optional, Union


GEOTIME_TYPE = Union[datetime, 'TimeInterval']
_DATE_FORMATS = [
    '%Y-%m-%dT%H:%M:%S.%f%z',
    '%Y-%m-%dT%H:%M:%S.%f',
    '%Y-%m-%dT%H:%M:%S%z',
    '%Y-%m-%d %H:%M:%S.%f%z',
    '%Y-%m-%d %H:%M:%S.%f',
    '%Y-%m-%d %H:%M:%S%z',
    '%Y-%m-%d',
]


class TimeInterval:
    """A class representing a right-open time interval"""

    def __init__(
        self,
        start: datetime = datetime.min,
        end: Union[datetime, timedelta] = datetime.max,
    ):
        """Check to make sure end is after start"""
        super().__init__()
        end = end if isinstance(end, datetime) else start + end

        if end < start:
            raise ValueError(f'end date {end} must not be less than start date {start}')

        self.start, self.end = self._default_to_zulu(start), self._default_to_zulu(end)

    def _default_to_zulu(self, dt: datetime) -> datetime:
        """Add Zulu/UTC as timezone, if timezone not present"""
        if not dt.tzinfo:
            # self.warn_once(
            #     'Datetime does not contain timezone information; Zulu/UTC time assumed. '
            #     '(this warning will not repeat)'
            # )
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
        """Create a TimeInterval from the FastKML equivalent."""
        from fastkml.times import TimeSpan, TimeStamp

        if isinstance(fastkml_time, TimeStamp):
            return TimeInterval(fastkml_time.timestamp.dt, fastkml_time.timestamp.dt)
        if isinstance(fastkml_time, TimeSpan):
            return TimeInterval(fastkml_time.begin.dt, fastkml_time.end.dt)

        raise ValueError('Unrecognized FastKML time object.')

    def _to_fastkml(self):
        """Convert to a FastKML equivalent object."""
        from fastkml.times import TimeSpan, TimeStamp, KmlDateTime

        if self.start == self.end:
            return TimeStamp(timestamp=KmlDateTime(dt=self.start))

        return TimeSpan(begin=KmlDateTime(self.start), end=KmlDateTime(self.end))

    def copy(self):
        return TimeInterval(self.start, self.end)

    def intersects(self, other: Union[datetime, TimeInterval]) -> bool:
        """Test whether this time interval intersects with another interval or point in time"""
        if isinstance(other, datetime):
            return other in self
        return not other.isdisjoint(self)

    def intersection(self, other: TimeInterval) -> Optional[TimeInterval]:
        """Returns a TimeInterval common to both time intervals. Will return None if impossible"""
        try:
            return TimeInterval(max(self.start, other.start), min(self.end, other.end))
        except ValueError:
            return None

    def isdisjoint(self, other: TimeInterval) -> bool:
        """Returns True if time intervals do not overlap."""
        if self.is_instant or other.is_instant:
            return self.end < other.start or self.start > other.end
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

    @classmethod
    def _get_timeformat(
        cls,
        time_str: str,
        formats: List[str] = _DATE_FORMATS
    ) -> str:
        for fmt in formats:
            try:
                datetime.strptime(time_str, fmt)
                return fmt
            except ValueError:
                continue
        raise ValueError(f'Date format was not recognized; {time_str}')

    @classmethod
    def from_str(
        cls,
        start: str,
        end: Optional[str] = None,
        time_format: Optional[Union[str, List[str]]] = None
    ) -> GEOTIME_TYPE:
        if time_format:
            formats = time_format if isinstance(time_format, list) else [time_format]
        if time_format is None:
            formats = _DATE_FORMATS

        fmt = cls._get_timeformat(start, formats)
        start = datetime.strptime(start, formats)
        end = start if end is None else datetime.strptime(end, fmt)

        return cls(start, end)

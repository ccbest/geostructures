"""Data structures for time intervals and milestones"""

from __future__ import annotations

__all__ = ['TimeInterval']

from datetime import date, datetime, timedelta, timezone
from typing import Optional, Union

from geostructures.utils.mixins import LoggingMixin, DefaultZuluMixin


class TimeInterval(LoggingMixin, DefaultZuluMixin):
    """A class representing a right-open time interval"""

    def __init__(
        self,
        start: datetime = datetime.min,
        end: Union[datetime, timedelta] = datetime.max,
    ):
        """Check to make sure end is after start"""
        super().__init__()
        end = end if isinstance(end, datetime) else start + end

        if end <= start:
            raise ValueError(f'end date {end} must not be less than start date {start}')

        self.start, self.end = self._default_to_zulu(start), self._default_to_zulu(end)

    def __eq__(self, other) -> bool:
        """Test equality"""
        if not isinstance(other, TimeInterval):
            return False

        return self.start == other.start and self.end == other.end

    def __repr__(self):
        """REPL representation"""
        return f'<TimeInterval [{self.start.isoformat()} - {self.end.isoformat()})>'

    def __contains__(self, dt: Union[datetime, TimeInterval]) -> bool:  # pylint: disable=invalid-name
        """Returns true if datetime is within range"""
        if isinstance(dt, datetime):
            return self.start <= self._default_to_zulu(dt) < self.end

        if isinstance(dt, TimeInterval):
            return self.issuperset(dt)

        raise ValueError('TimeIntervals may only contain datetimes and other TimeIntervals.')

    def __hash__(self) -> int:
        return hash((self.start, self.end))

    def isdisjoint(self, other: TimeInterval) -> bool:
        """Returns True if time intervals do not overlap."""
        return self.end <= other.start or self.start >= other.end

    def issubset(self, other: TimeInterval) -> bool:
        """Returns True if time interval is contained entirely within other interval"""
        return (other.start <= self.start) and (self.end <= other.end)

    def issuperset(self, other: TimeInterval) -> bool:
        """Returns True if other interval is contained entirely within time interval"""
        return other.issubset(self)

    def elapsed(self):
        """Returns elapsed time (as a timedelta) for the interval"""
        return self.end - self.start

    def union(self, other: TimeInterval) -> TimeInterval:
        """Returns a TimeInterval that spans both time intervals"""
        return TimeInterval(min(self.start, other.start), max(self.end, other.end))

    def intersection(self, other: TimeInterval) -> Optional[TimeInterval]:
        """Returns a TimeInterval common to both time intervals. Will return None if impossible"""
        try:
            return TimeInterval(max(self.start, other.start), min(self.end, other.end))
        except ValueError:
            return None

"""Data structures for time intervals and milestones"""

from __future__ import annotations

__all__ = ['DateInterval', 'TimeInterval']

from datetime import date, datetime, timedelta
from typing import Optional, Union

from geostructures.utils.mixins import LoggingMixin, DefaultZuluMixin


class DateInterval(LoggingMixin, DefaultZuluMixin):

    """
    An interval representing N contiguous dates, inclusive.

    Args:
        start:  datetime.date
            The starting date of the date interval. If not passed, behaves as all dates in the past.

        end:  Union[datetime.date, datetime.timedelta, int]
            The last date of the date interval, inclusive. If not passed, behaves as all
            ates in the future.

            You can provide any of:
                - A `Date`, indicating the end date
                - A `timedelta`, indicating the number of days from the `start`
                - A `int`, indicating the number of days from the `start`
    """

    def __init__(
        self,
        start: date = date.min,
        end: Union[date, timedelta, int] = date.max,
    ):
        super().__init__()
        self.start = start

        if isinstance(end, date):
            self.end = end

        elif isinstance(end, int):
            self.end = start + timedelta(days=end)

        elif isinstance(end, timedelta):
            if end.seconds or end.microseconds:
                self.logger.warning(
                    'Time increments smaller than a day are not supported by DateInterval (see '
                    'TimeInterval). Your timedelta will be truncated to the nearest whole '
                    'number of days: %d',
                    end.days,
                )
            end = timedelta(days=end.days)
            self.end = start + end

        if self.end < self.start:
            raise ValueError(
                f'end date {end} must not be less than or equal to start date {start}'
            )

    def __contains__(self, d: date) -> bool:
        """Returns True if date is within range"""
        return self.start <= d <= self.end

    def __eq__(self, other) -> bool:
        """Test equality"""
        if not isinstance(other, DateInterval):
            return False

        return self.start == other.start and self.end == other.end

    def __hash__(self) -> int:
        """Object hashing"""
        return hash((self.start, self.end))

    def __repr__(self):
        """REPL representation"""
        return (
            f'<DateInterval [{self.start.strftime("%Y-%m-%d")} - {self.end.strftime("%Y-%m-%d")}]>'
        )

    def isdisjoint(self, other: DateInterval) -> bool:
        """Returns True if date intervals do not overlap."""
        return self.end < other.start or self.start > other.end

    def issubset(self, other: DateInterval) -> bool:
        """Returns True if date interval is contained entirely within other interval"""
        return (other.start <= self.start) and (other.end >= self.end)

    def issuperset(self, other: DateInterval) -> bool:
        """Returns True if other interval is contained entirely within date interval"""
        return other.issubset(self)

    @property
    def elapsed(self):
        """Returns elapsed time (as a timedelta) for the interval"""
        return self.end - self.start + timedelta(days=1)

    def union(self, other: DateInterval) -> DateInterval:
        """Returns a DateInterval that spans both time intervals"""
        min_start = min(self.start, other.start)
        max_end = max(self.end, other.end)
        return DateInterval(min_start, max_end)

    def intersection(self, other: DateInterval) -> Optional[DateInterval]:
        """Returns a DateInterval common to both date intervals. Will return None if impossible"""
        max_start = max(self.start, other.start)
        min_end = min(self.end, other.end)
        return None if min_end < max_start else DateInterval(max_start, min_end)


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

    def __contains__(self, dt: datetime) -> bool:  # pylint: disable=invalid-name
        """Returns true if datetime is within range"""
        return self.start <= self._default_to_zulu(dt) < self.end

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

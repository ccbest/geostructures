"""Utility mixin classes"""

__all__ = ['DefaultZuluMixin', 'WarnOnceMixin']

from datetime import datetime, timezone

from geostructures import LOGGER


class WarnOnceMixin:
    """
    Adds a private method for converting a datetime to UTC if no timezone is present.

    """
    WARNED_ONCE: set = set()

    def __init__(self):
        self.logger = LOGGER

    @classmethod
    def _set_warned_once(cls, msg):
        """Appends message to classvar"""
        cls.WARNED_ONCE.add(msg)

    def warn_once(self, msg, *args, **kwargs):
        """Logs a warning only once per message"""
        if msg in self.WARNED_ONCE:
            return

        self.logger.warning(msg, *args, **kwargs)
        self._set_warned_once(msg)


class DefaultZuluMixin(WarnOnceMixin):  # pylint: disable=too-few-public-methods

    def _default_to_zulu(self, dt: datetime) -> datetime:
        """Add Zulu/UTC as timezone, if timezone not present"""
        if not dt.tzinfo:
            self.warn_once(
                'Datetime does not contain timezone information; Zulu/UTC time assumed. '
                '(this warning will not repeat)'
            )
            return dt.replace(tzinfo=timezone.utc)

        return dt

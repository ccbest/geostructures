"""Utility mixin classes"""

__all__ = ['LoggingMixin', 'DefaultZuluMixin']

from datetime import datetime, timezone
import logging
from typing import Optional


class LoggingMixin:  # pylint: disable=too-few-public-methods
    """Mixin class for logging"""
    logger: logging.Logger

    WARNED_ONCE: set = set()

    def __init__(self, logstr: Optional[str] = None):
        _class = self.__class__
        module_name = _class.__module__
        classname = _class.__name__
        if logstr:
            classname += f'.{logstr}'

        logstr = f"{classname}" if module_name == "builtins" else f"{module_name}.{classname}"

        self.logger = logging.getLogger(logstr)

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


class DefaultZuluMixin:  # pylint: disable=too-few-public-methods

    """
    Adds a private method for converting a datetime to UTC if no timezone is present.

    """
    WARNED_ONCE: set = set()

    def __init__(self):
        self.logger = logging.getLogger()

    def _default_to_zulu(self, dt: datetime) -> datetime:
        """Add Zulu/UTC as timezone, if timezone not present"""
        if not dt.tzinfo:
            self.warn_once(
                'Datetime does not contain timezone information; Zulu/UTC time assumed. '
                '(this warning will not repeat)'
            )
            return dt.replace(tzinfo=timezone.utc)

        return dt

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

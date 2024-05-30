"""Logging utility for geostructures"""

__all__ = ['LOGGER', 'warn_once']

import logging

LOGGER = logging.getLogger('geostructures')
LOGGER.setLevel(logging.WARNING)
_LOG_HANDLER = logging.StreamHandler()
_LOG_FORMATTER = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
_LOG_HANDLER.setFormatter(_LOG_FORMATTER)
LOGGER.addHandler(_LOG_HANDLER)

_WARNINGS = set()


def warn_once(warning: str):
    if warning not in _WARNINGS:
        LOGGER.warning(warning)
        _WARNINGS.add(warning)

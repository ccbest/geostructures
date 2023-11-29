"""Logging utility for geostructures"""
__all__ = ['LOGGER']

import logging

LOGGER = logging.getLogger('geostructures')
LOGGER.setLevel(logging.WARNING)
_LOG_HANDLER = logging.StreamHandler()
_LOG_FORMATTER = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
_LOG_HANDLER.setFormatter(_LOG_FORMATTER)
LOGGER.addHandler(_LOG_HANDLER)

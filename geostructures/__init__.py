
import sys

from geostructures._version import __version__  # noqa: F401
from geostructures.utils.logging import LOGGER
from geostructures.coordinates import Coordinate
from geostructures.structures import (
    GeoBox, GeoCircle, GeoEllipse, GeoLineString, GeoPoint, GeoPolygon,
    GeoRing
)
from geostructures.multistructures import MultiGeoLineString, MultiGeoPoint, MultiGeoPolygon
from geostructures.collections import FeatureCollection, Track
from geostructures.utils.conditional_imports import ConditionalPackageInterceptor


ConditionalPackageInterceptor.permit_packages(
    {
        'geopandas': 'geostructures[df]',
        'h3': 'geostructures[h3]',
        'fastkml': 'geostructures[kml]',
        'mgrs': 'geostructures[mgrs]',
        'pandas': 'geostructures[df]',
        'pyproj': 'geostructures[proj]',
    }
)
sys.meta_path.append(ConditionalPackageInterceptor)  # type: ignore

__all__ = [
    'Coordinate',
    'FeatureCollection',
    'GeoBox',
    'GeoCircle',
    'GeoEllipse',
    'GeoLineString',
    'GeoPoint',
    'GeoPolygon',
    'GeoRing',
    'MultiGeoLineString',
    'MultiGeoPoint',
    'MultiGeoPolygon',
    'Track',
    'LOGGER',
]

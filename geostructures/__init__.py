
import sys

from geostructures._version import __version__  # noqa: F401
from geostructures.utils.logging import LOGGER
from geostructures.coordinates import Coordinate
from geostructures.structures import (
    GeoBox, GeoCircle, GeoEllipse, GeoLineString, GeoPoint, GeoPolygon,
    GeoRing
)
from geostructures.multistructures import MultiGeoLineString, MultiGeoPoint, MultiGeoShape
from geostructures.collections import FeatureCollection, Track
from geostructures.utils.conditional_imports import ConditionalPackageInterceptor


ConditionalPackageInterceptor.permit_packages(
    {
        'geopandas': 'geopandas>=0.13,<1',
        'h3': 'h3>=3.7,<4',
        'mgrs': 'mgrs>=1.4.5,<2',
        'pandas': 'pandas>=2,<3',
        'plotly': 'plotly>=5,<6',
        'pyproj': 'pyproj>=3.6,<4',
        'scipy': 'scipy>=3.0.7,<4.0',
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
    'MultiGeoShape',
    'Track',
    'LOGGER',
]

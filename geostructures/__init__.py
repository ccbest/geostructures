
import sys

from geostructures.utils.logging import LOGGER
from geostructures.coordinates import Coordinate
from geostructures.structures import (
    GeoBox, GeoCircle, GeoEllipse, GeoLineString, GeoPoint, GeoPolygon,
    GeoRing
)
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
        'fastkml': 'fastkml>=0.12.0,<1.0',
    }
)
sys.meta_path.append(ConditionalPackageInterceptor)  # type: ignore

__all__ = [
    'Coordinate',
    'GeoBox',
    'GeoCircle',
    'GeoEllipse',
    'GeoLineString',
    'GeoPoint',
    'GeoPolygon',
    'GeoRing',
    'LOGGER',
]

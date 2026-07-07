"""
Helpers for optional third-party dependencies
"""

__all__ = ['import_optional']

import importlib


# Maps a package's import name to the geostructures extra that provides it
_PACKAGE_EXTRAS = {
    'fastkml': 'kml',
    'geographiclib': 'karney',
    'geopandas': 'df',
    'h3': 'h3',
    'mgrs': 'mgrs',
    'pandas': 'df',
    'pyproj': 'proj',
    'shapefile': 'shapefile',
    'shapely': 'shapely',
}


def import_optional(package: str):
    """
    Import an optional dependency, raising an ImportError that names the
    geostructures extra providing it if it is not installed.

    Args:
        package:
            The importable package name, e.g. 'h3'

    Returns:
        The imported module
    """
    try:
        return importlib.import_module(package)
    except ImportError as exc:
        extra = _PACKAGE_EXTRAS.get(package.split('.', maxsplit=1)[0])
        install_target = f'geostructures[{extra}]' if extra else package
        raise ImportError(
            f'The optional dependency {package!r} is required for this feature. '
            'Please execute the following command to continue:\n'
            f'    pip install {install_target}'
        ) from exc

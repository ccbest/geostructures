"""Module for serializing geostructures out to external structures.

This module is the write-side counterpart to :mod:`geostructures.parsers`:
where ``parse_*`` reads an external representation into geostructures,
``serialize_*`` renders geostructures back out to that representation.
"""

__all__ = [
    'serialize_geojson',
    'serialize_kml',
    'serialize_shapefile',
    'serialize_wkt',
]

import json
from io import BytesIO
from pathlib import Path
from typing import IO, BinaryIO, List, Optional, Union
from zipfile import ZIP_DEFLATED, ZipFile

from geostructures._base import BaseShape
from geostructures.collections import CollectionBase, FeatureCollection
from geostructures.utils.conditional_imports import import_optional

# The conventional name for the primary KML document inside a KMZ archive
_KMZ_MAIN_DOCUMENT = 'doc.kml'


def _emit_bytes(payload: bytes, file: Optional[Union[str, Path, BinaryIO]]) -> Optional[bytes]:
    """Return the bytes when *file* is None, else write them to the path/stream."""
    if file is None:
        return payload
    if isinstance(file, (str, Path)):
        Path(file).write_bytes(payload)
        return None
    if hasattr(file, 'write'):
        file.write(payload)
        return None
    raise TypeError(f'Cannot serialize to object of type {type(file).__name__!r}.')


def _emit_text(text: str, file: Optional[Union[str, Path, IO]], encoding: str) -> Optional[str]:
    """Return the text when *file* is None, else write it to the path/stream."""
    if file is None:
        return text
    if isinstance(file, (str, Path)):
        Path(file).write_text(text, encoding=encoding)
        return None
    if hasattr(file, 'write'):
        file.write(text)
        return None
    raise TypeError(f'Cannot serialize to object of type {type(file).__name__!r}.')


def serialize_kml(
    data: Union[CollectionBase, BaseShape],
    file: Optional[Union[str, Path, BinaryIO]] = None,
    *,
    kmz: Optional[bool] = None,
    encoding: str = 'utf8',
    folder_name: str = 'geostructures',
) -> Optional[bytes]:
    """
    Serialize geostructures to a KML/KMZ document - the inverse of ``parse_kml``.

    Whether a plain KML document or a zipped KMZ archive is produced is inferred
    from the ``file`` extension (``.kmz`` -> KMZ, anything else -> KML), matching
    how ``parse_kml`` dispatches on a file's suffix. Pass ``kmz`` explicitly to
    override the inference, e.g. when writing to a stream or returning bytes.

    Args:
        data:
            A collection (FeatureCollection or Track) or a single GeoShape to
            serialize. A single shape is wrapped in a FeatureCollection.

        file: (Optional)
            Where to write the output. May be a path (str or pathlib.Path) or a
            binary file-like object. If omitted, the serialized bytes are
            returned instead of written.

        kmz: (Optional)
            If provided, forces KMZ output (True) or plain KML output (False),
            overriding the extension-based inference.

        encoding:
            The text encoding used to encode the KML document (default utf8).

        folder_name:
            The name of the KML Folder the shapes are placed in (default
            'geostructures'). Note that ``parse_kml`` records folder names as
            ``sub_folder_N`` properties for traceability, so a round trip adds
            this key to each shape's properties.

    Returns:
        bytes if ``file`` is None, otherwise None (the output is written to
        ``file``).
    """
    import_optional('fastkml')
    from fastkml import KML, Document

    if isinstance(data, BaseShape):
        data = FeatureCollection([data])

    # Reuse the existing object-level serialization: to_fastkml_folder builds a
    # Folder of Placemarks (geometry + ExtendedData properties + times).
    folder = data.to_fastkml_folder(folder_name)
    document = Document(features=[folder])
    kml_text = KML(features=[document]).to_string()

    if kmz is None:
        kmz = isinstance(file, (str, Path)) and Path(file).suffix.lower() == '.kmz'

    if kmz:
        payload = _to_kmz_bytes(kml_text, encoding)
    else:
        payload = kml_text.encode(encoding)

    return _emit_bytes(payload, file)


def _to_kmz_bytes(kml_text: str, encoding: str) -> bytes:
    """Zip a KML document into KMZ archive bytes (as a single ``doc.kml`` member)."""
    buffer = BytesIO()
    with ZipFile(buffer, 'w', ZIP_DEFLATED) as archive:
        archive.writestr(_KMZ_MAIN_DOCUMENT, kml_text.encode(encoding))
    return buffer.getvalue()


def serialize_geojson(
    data: Union[CollectionBase, BaseShape],
    file: Optional[Union[str, Path, IO]] = None,
    *,
    indent: Optional[int] = None,
    encoding: str = 'utf8',
    **kwargs,
) -> Optional[str]:
    """
    Serialize geostructures to a GeoJSON document - the inverse of ``parse_geojson``.

    A single shape serializes to a GeoJSON ``Feature``; a collection to a
    ``FeatureCollection``. Reuses the existing ``to_geojson`` methods and encodes
    the resulting object to a JSON string.

    Args:
        data:
            A collection (FeatureCollection or Track) or a single GeoShape.

        file: (Optional)
            Where to write the output - a path (str or pathlib.Path) or a text
            file-like object. If omitted, the JSON string is returned.

        indent: (Optional)
            If provided, pretty-print the JSON with this indent width.

        encoding:
            The text encoding used when writing to a path (default utf8).

        kwargs:
            Forwarded to ``to_geojson`` (e.g. ``properties``, ``k``).

    Returns:
        str if ``file`` is None, otherwise None.
    """
    text = json.dumps(data.to_geojson(**kwargs), indent=indent)
    return _emit_text(text, file, encoding)


def serialize_wkt(
    data: Union[CollectionBase, BaseShape],
    file: Optional[Union[str, Path, IO]] = None,
    *,
    encoding: str = 'utf8',
    **kwargs,
) -> Optional[str]:
    """
    Serialize a single shape to a WKT string - the inverse of ``parse_wkt``.

    WKT carries only geometry (no properties or time) and represents a single
    geometry, so collections are rejected: use ``serialize_geojson`` for those.

    Args:
        data:
            A single GeoShape (including multi-shapes).

        file: (Optional)
            Where to write the output - a path (str or pathlib.Path) or a text
            file-like object. If omitted, the WKT string is returned.

        encoding:
            The text encoding used when writing to a path (default utf8).

        kwargs:
            Forwarded to ``to_wkt`` (e.g. ``k`` for curved shapes).

    Returns:
        str if ``file`` is None, otherwise None.
    """
    if isinstance(data, CollectionBase):
        raise TypeError(
            'serialize_wkt expects a single shape; WKT cannot represent a '
            'collection. Use serialize_geojson for collections.'
        )
    return _emit_text(data.to_wkt(**kwargs), file, encoding)


def serialize_shapefile(
    data: Union[CollectionBase, BaseShape],
    file: Optional[Union[str, Path, BinaryIO]] = None,
    *,
    include_properties: Optional[List[str]] = None,
) -> Optional[bytes]:
    """
    Serialize geostructures to a zip-archived ESRI shapefile set - the inverse of
    ``parse_shapefile``.

    Reuses the existing ``CollectionBase.to_shapefile`` writer, wrapping its
    open-ZipFile requirement so callers can supply a path (matching how
    ``parse_shapefile``/``from_shapefile`` accept one), a binary stream, or
    nothing (to receive the archive bytes).

    Args:
        data:
            A collection (FeatureCollection or Track) or a single GeoShape.

        file: (Optional)
            Where to write the archive - a path (str or pathlib.Path) or a binary
            file-like object. If omitted, the zip archive bytes are returned.

        include_properties: (Optional)
            The property keys to include as shapefile fields. Defaults to all.

    Returns:
        bytes if ``file`` is None, otherwise None.
    """
    import_optional('shapefile')

    if isinstance(data, BaseShape):
        data = FeatureCollection([data])

    # No destination: build the archive in memory and return its bytes.
    if file is None:
        buffer = BytesIO()
        with ZipFile(buffer, 'w', ZIP_DEFLATED) as archive:
            data.to_shapefile(archive, include_properties=include_properties)
        return buffer.getvalue()

    # zipfile.ZipFile accepts a path (str/Path) or a binary file-like object
    if not isinstance(file, (str, Path)) and not hasattr(file, 'write'):
        raise TypeError(f'Cannot serialize shapefile to object of type {type(file).__name__!r}.')

    with ZipFile(file, 'w', ZIP_DEFLATED) as archive:
        data.to_shapefile(archive, include_properties=include_properties)
    return None

"""Module for serializing geostructures out to external structures.

This module is the write-side counterpart to :mod:`geostructures.parsers`:
where ``parse_*`` reads an external representation into geostructures,
``serialize_*`` renders geostructures back out to that representation.
"""

__all__ = [
    'serialize_kml'
]

from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Optional, Union
from zipfile import ZIP_DEFLATED, ZipFile

from geostructures._base import BaseShape
from geostructures.collections import CollectionBase, FeatureCollection
from geostructures.utils.conditional_imports import import_optional

# The conventional name for the primary KML document inside a KMZ archive
_KMZ_MAIN_DOCUMENT = 'doc.kml'


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

    if file is None:
        return payload

    if isinstance(file, (str, Path)):
        Path(file).write_bytes(payload)
        return None

    if hasattr(file, 'write'):
        file.write(payload)
        return None

    raise TypeError(f'Cannot serialize KML to object of type {type(file).__name__!r}.')


def _to_kmz_bytes(kml_text: str, encoding: str) -> bytes:
    """Zip a KML document into KMZ archive bytes (as a single ``doc.kml`` member)."""
    buffer = BytesIO()
    with ZipFile(buffer, 'w', ZIP_DEFLATED) as archive:
        archive.writestr(_KMZ_MAIN_DOCUMENT, kml_text.encode(encoding))
    return buffer.getvalue()

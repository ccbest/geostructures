"""Module for parsing external structures into geostructures"""

__all__ = [
    'parse_geojson', 'parse_kml', 'parse_wkt'
]

import json
from io import BytesIO
from pathlib import Path
import re
from typing import cast, Any, Dict, List, Optional, Union
from zipfile import ZipFile

from geostructures.collections import FeatureCollection
from geostructures.structures import GeoPolygon, GeoPoint, GeoLineString
from geostructures.multistructures import MultiGeoPoint, MultiGeoPolygon, MultiGeoLineString
from geostructures.typing import GeoShape, SimpleShape
from geostructures.utils.conditional_imports import import_optional
from geostructures.utils.logging import warn_once


_PARSER_MAP: Dict[str, SimpleShape] = {
    'POINT': GeoPoint,
    'LINESTRING': GeoLineString,
    'POLYGON': GeoPolygon,
    'MULTIPOINT': MultiGeoPoint,
    'MULTILINESTRING': MultiGeoLineString,
    'MULTIPOLYGON': MultiGeoPolygon,
}

_TYPE_RE = re.compile(r'\s*([A-Za-z]+)')

# ZIP archives - and therefore KMZ files - always begin with this magic number
_KMZ_MAGIC = b'PK\x03\x04'


def _parse_fastkml(
    kml,
    _shapes: Optional[List[GeoShape]] = None,
    _depth: int = 0,
    _props: Optional[Dict[str, str]] = None,
):
    """
    Recurses through a FastKML object tree, extracting all Placemarks and
    converting them into their corresponding geostructures.

    This is the low-level, in-memory workhorse behind the public
    ``parse_kml``; most callers should use ``parse_kml`` instead.

    Args:
        kml:
            A FastKML.KML object

        _shapes: (List[GeoShape])
            Internal use only. Mutated with geostructures as they're
            extracted from the KML

        _depth: (int)
            Internal use only. The recursion depth.

        _props: (Dict[str, str])
            Internal use only. Information about higher-level containers
            (e.g. folder names) to store as properties on the shape
            for traceability.

    Returns:
        List[GeoShape]
    """
    import_optional('fastkml')
    from fastkml import KML, Document, Folder, Placemark

    _shapes = _shapes if _shapes is not None else []
    _props = _props if _props is not None else {}

    if isinstance(kml, (KML, Document, Folder)):
        # Recurse
        if isinstance(kml, Folder):
            # Inject subfolder name into props for traceability. Use a copy so
            # the folder name does not leak to siblings outside this folder
            _props = {**_props, f'sub_folder_{_depth}': kml.name or 'Unnamed Folder'}

        for feature in kml.features:
            _parse_fastkml(feature, _shapes, _depth + 1, _props)

        return _shapes

    if isinstance(kml, Placemark):
        # Parse the shape and mutate _shapes
        if kml.geometry is None:
            # It's possible to create a placeless placemark
            return _shapes

        geom_type = kml.geometry.__geo_interface__['type'].upper()
        parser = _PARSER_MAP.get(geom_type)
        if parser is None:
            # e.g. a heterogeneous MultiGeometry surfaces as a GeometryCollection,
            # which has no single geostructures equivalent. Skip it rather than
            # crashing the entire parse.
            warn_once(f'Skipping KML placemark with unsupported geometry type {geom_type!r}.')
            return _shapes

        shape = parser.from_fastkml_placemark(kml)  # type: ignore
        for key, value in _props.items():
            shape.set_property(key, value, inplace=True)
        for prop in ('name', 'description', 'address', 'phone_number'):
            # Inject KML properties into shape properties
            if getattr(kml, prop) is not None:
                shape.set_property(prop, getattr(kml, prop), inplace=True)
        _shapes.append(shape)
        return _shapes

    return _shapes  # pragma: no cover


def parse_geojson(
    gjson: Union[str, Dict[str, Any]],
    time_start_property: str = 'datetime_start',
    time_end_property: str = 'datetime_end',
    time_format: Optional[str] = None,
):
    """
    Parses a GeoJSON structure into its corresponding geostructure(s).

    Args:
        gjson:
            A GeoJSON structure (as a string or python dict)

        time_start_property:
            The geojson property containing the start time, if available

        time_end_property:
            The geojson property containing hte ned time, if available

        time_format: (Optional)
            The format of the timestamps in the above time fields.

    Returns:
        GeoShape, subtype determined by input
    """
    PARSER_MAP = {
        **_PARSER_MAP,
        'FEATURECOLLECTION': FeatureCollection
    }

    if isinstance(gjson, str):
        gjson = json.loads(gjson)

    gjson = cast(Dict[str, Any], gjson)

    parser = None
    if 'type' in gjson and gjson['type'].upper() in PARSER_MAP:
        parser = PARSER_MAP[gjson['type'].upper()]

    elif gjson.get('geometry', {}).get('type', '').upper() in PARSER_MAP:
        parser = PARSER_MAP[gjson['geometry']['type'].upper()]

    if not parser:
        raise ValueError('Failed to parse geojson.')

    return parser.from_geojson(  # type: ignore
        gjson,
        time_start_property,
        time_end_property,
        time_format
    )


def parse_wkt(wkt: str, /, **kwargs):
    """
    Convert a WKT string to its corresponding geostructure.

    Extra keyword arguments are forwarded to the *.from_wkt()* call so you
    can supply *dt* or *properties* if desired.
    """
    m = _TYPE_RE.match(wkt)
    if not m:
        raise ValueError('Invalid WKT. could not find geometry keyword.')

    geom_type = m.group(1).upper()
    try:
        parser = _PARSER_MAP[geom_type]
    except KeyError as exc:
        raise ValueError(f'Unsupported WKT geometry {geom_type}.') from exc

    return parser.from_wkt(wkt, **kwargs)                 # type: ignore[arg-type]


def parse_kml(
    data: Union[str, bytes, Path, Any],
    encoding: str = 'utf8'
) -> FeatureCollection:
    """
    Parse KML/KMZ input into a FeatureCollection, detecting the input type.

    Accepts any of:
        - An in-memory FastKML object (KML, Document, Folder, or Placemark)
        - A path (str or pathlib.Path) to a .kml or .kmz file
        - A raw KML document, as a str or bytes
        - Raw KMZ archive bytes

    Shapes read from a file are tagged with ``filepath`` (and ``filename``)
    properties for traceability.

    Args:
        data:
            The KML input, in any of the forms above.

        encoding:
            The text encoding used to decode raw or file bytes (default utf8).

    Returns:
        FeatureCollection
    """
    import_optional('fastkml')
    from fastkml import KML, Document, Folder, Placemark

    # 1. An already-parsed, in-memory FastKML object
    if isinstance(data, (KML, Document, Folder, Placemark)):
        return FeatureCollection(_parse_fastkml(data))

    # 2. Raw bytes - either a KMZ archive or an encoded KML document
    if isinstance(data, bytes):
        if data.startswith(_KMZ_MAGIC):
            return FeatureCollection(_parse_kmz(BytesIO(data), encoding))
        return FeatureCollection(_parse_kml_bytes(data, encoding))

    # 3. A raw KML document passed as text (as opposed to a file path)
    if isinstance(data, str) and data.lstrip().startswith('<'):
        return FeatureCollection(_parse_kml_bytes(data.encode(encoding), encoding))

    # 4. A filesystem path to a .kml or .kmz file
    if isinstance(data, (str, Path)):
        return _read_kml_file(Path(data), encoding)

    raise TypeError(f'Cannot parse KML from object of type {type(data).__name__!r}.')


def _parse_kml_bytes(content: bytes, encoding: str) -> List[GeoShape]:
    """Parse a raw KML document (bytes) into a list of GeoShapes."""
    import_optional('fastkml')
    from fastkml import KML

    # fastkml's from_string expects text; decode with the caller-provided encoding.
    kml = KML.from_string(content.decode(encoding))
    return _parse_fastkml(kml)


def _parse_kmz(source, encoding: str, filepath: Optional[str] = None) -> List[GeoShape]:
    """
    Parse every .kml member of a KMZ archive into a list of GeoShapes.

    Args:
        source:
            A path or file-like object accepted by zipfile.ZipFile.

        encoding:
            The text encoding used to decode each member.

        filepath: (Optional)
            When reading from disk, the archive path to tag onto each shape
            (as a str, so the property stays JSON-serializable). Omitted when
            parsing raw bytes, which have no path.

    Returns:
        List[GeoShape]
    """
    shapes: List[GeoShape] = []
    with ZipFile(source, 'r') as z:
        # Filter for .kml files inside the zip
        kml_files = [x for x in z.namelist() if x.lower().endswith('.kml')]
        for kml_f in kml_files:
            parsed = _parse_kml_bytes(z.read(kml_f), encoding)
            for shape in parsed:
                if filepath is not None:
                    shape.set_property('filepath', filepath)
                shape.set_property('filename', kml_f)
            shapes.extend(parsed)
    return shapes


def _read_kml_file(fpath: Path, encoding: str) -> FeatureCollection:
    """
    Read a .kml or .kmz file from disk and parse it into a FeatureCollection.
    If a KMZ is provided, all KML files within it are parsed and combined.
    """
    if not fpath.exists():
        raise FileNotFoundError(fpath)

    if fpath.suffix.lower() == '.kmz':
        shapes = _parse_kmz(fpath, encoding, filepath=str(fpath))
    else:
        # Assume plain KML
        shapes = _parse_kml_bytes(fpath.read_bytes(), encoding)
        for shape in shapes:
            # Store the path as a str so the property stays JSON-serializable
            shape.set_property('filepath', str(fpath))
            shape.set_property('filename', fpath.name)

    return FeatureCollection(shapes)

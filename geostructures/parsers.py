"""Module for parsing external structures into geostructures"""

__all__ = [
    'parse_fastkml', 'parse_geojson', 'parse_wkt'
]

import json
import re
from typing import cast, Any, Dict, List, Optional, Union

from geostructures.collections import FeatureCollection
from geostructures.structures import GeoPolygon, GeoPoint, GeoLineString
from geostructures.multistructures import MultiGeoPoint, MultiGeoPolygon, MultiGeoLineString
from geostructures.typing import GeoShape, SimpleShape


_PARSER_MAP: Dict[str, SimpleShape] = {
    'POINT': GeoPoint,
    'LINESTRING': GeoLineString,
    'POLYGON': GeoPolygon,
    'MULTIPOINT': MultiGeoPoint,
    'MULTILINESTRING': MultiGeoLineString,
    'MULTIPOLYGON': MultiGeoPolygon,
}


def parse_fastkml(
    kml,
    _shapes: Optional[List[GeoShape]] = None,
    _depth: int = 0,
    _props: Optional[Dict[str, str]] = None,
):
    """
    Recurses through a KML, extracting all Placemarks and converting
    them into their corresponding geostructures.

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
    from fastkml import KML, Document, Folder, Placemark

    _shapes = _shapes if _shapes is not None else []
    _props = _props if _props is not None else {}

    if isinstance(kml, (KML, Document, Folder)):
        # Recurse
        if isinstance(kml, Folder):
            # Inject subfolder name into props for traceability
            _props[f'sub_folder_{_depth}'] = kml.name or 'Unnamed Folder'

        for feature in kml.features:
            parse_fastkml(feature, _shapes, _depth + 1, _props)

        return _shapes

    if isinstance(kml, Placemark):
        # Parse the shape and mutate _shapes
        if kml.geometry is None:
            # It's possible to create a placeless placemark
            return _shapes

        parser = _PARSER_MAP[kml.geometry.__geo_interface__['type'].upper()]  # type: ignore
        shape = parser.from_fastkml_placemark(kml)
        props = shape._properties or {}
        props.update(_props)
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


def parse_wkt(wkt: str) -> GeoShape:
    """
    Parses a WKT string into its corresponding geostructure.

    Args:
        wkt: (str)
            A well known text string, representing a simplified geometry

    Returns:
        GeoShape, subtype determined by input
    """
    wkt_type_match = re.match(r'^[a-zA-Z]+', wkt)
    if wkt_type_match is None:
        raise ValueError('Invalid WKT')

    wkt_type = wkt_type_match.group()
    if wkt_type not in _PARSER_MAP:
        raise ValueError('Invalid WKT.')

    return _PARSER_MAP[wkt_type].from_wkt(wkt)  # type: ignore

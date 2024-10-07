
import json
import re
from typing import cast, Any, Dict, Optional, Union

from geostructures.collections import FeatureCollection
from geostructures.structures import GeoPolygon, GeoPoint, GeoLineString
from geostructures.multistructures import MultiGeoPoint, MultiGeoPolygon, MultiGeoLineString
from geostructures.typing import GeoShape


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
    if isinstance(gjson, str):
        gjson = json.loads(gjson)

    gjson = cast(Dict[str, Any], gjson)
    parser_map = {
        'POINT': GeoPoint,
        'LINESTRING': GeoLineString,
        'POLYGON': GeoPolygon,
        'MULTIPOINT': MultiGeoPoint,
        'MULTILINESTRING': MultiGeoLineString,
        'MULTIPOLYGON': MultiGeoPolygon,
        'FEATURECOLLECTION': FeatureCollection
    }
    parser = None
    if 'type' in gjson and gjson['type'].upper() in parser_map:
        parser = parser_map[gjson['type'].upper()]

    elif gjson.get('geometry', {}).get('type', '').upper() in parser_map:
        parser = parser_map[gjson['geometry']['type'].upper()]

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
    parser_map = {
        'POINT': GeoPoint,
        'LINESTRING': GeoLineString,
        'POLYGON': GeoPolygon,
        'MULTIPOINT': MultiGeoPoint,
        'MULTILINESTRING': MultiGeoLineString,
        'MULTIPOLYGON': MultiGeoPolygon,
    }
    wkt_type = re.split(r'\s?\(', wkt, 1)[0].upper()
    if wkt_type not in parser_map:
        raise ValueError('Invalid WKT.')

    return parser_map[wkt_type].from_wkt(wkt)  # type: ignore


import re
from typing import Any, Dict, Optional

from geostructures.collections import FeatureCollection
from geostructures.structures import GeoPolygon, GeoPoint, GeoLineString
from geostructures.multistructures import MultiGeoPoint, MultiGeoPolygon, MultiGeoLineString


def parse_geojson(
    gjson: Dict[str, Any],
    time_start_property: str = 'datetime_start',
    time_end_property: str = 'datetime_end',
    time_format: Optional[str] = None,
):
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


def parse_wkt(wkt: str):
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

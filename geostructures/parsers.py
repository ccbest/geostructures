"""Module for parsing external structures into geostructures"""

__all__ = [
    'parse_fastkml', 'parse_geojson', 'parse_wkt'
]

import json
import re
from typing import cast, Any, Dict, List, Optional, Union

from geostructures.collections import FeatureCollection
from geostructures.coordinates import Coordinate
from geostructures.structures import GeoPolygon, GeoPoint, GeoLineString
from geostructures.multistructures import MultiGeoPoint, MultiGeoPolygon, MultiGeoLineString
from geostructures.typing import GeoShape, SimpleShape


_PARSER_MAP: Dict[str, SimpleShape] = {
    'POINT': GeoPoint,
    'POINTGEOMETRY': GeoPoint,
    'LINESTRING': GeoLineString,
    'POLYGON': GeoPolygon,
    'MULTIPOINT': MultiGeoPoint,
    'MULTILINESTRING': MultiGeoLineString,
    'MULTIPOLYGON': MultiGeoPolygon,
}


def _get_datetime_pandas(start_time, end_time):
    """
    Converts pandas Timestamps to Python datetime objects and returns a TimeInterval.

    Args:
        start_time (pd.Timestamp or None): The start time.
        end_time (pd.Timestamp or None): The end time.

    Returns:
        TimeInterval: The time interval representing the start and end time.
    """
    import pandas as pd
    from geostructures.time import TimeInterval

    if pd.notnull(start_time) or pd.notnull(end_time):
        if isinstance(start_time, pd.Timestamp):
            start_time = start_time.to_pydatetime()

        if isinstance(end_time, pd.Timestamp):
            end_time = end_time.to_pydatetime()

        return TimeInterval(start_time, end_time)


def parse_arcgis_featureclass(
    row,
    columns,
    time_start_property: Optional[Union[str, datetime]] = None,
    time_end_property: Optional[Union[str, datetime]] = None,
    time_fmt: Optional[Union[str, List[str]]] = None,
): -> GeoShape
    """
    Parses an ArcGIS feature class row into a geospatial structure.

    Args:
        row:
            The row from an ArcGIS feature class, typically obtained using an arcgis GeoAccessor.

        columns:
            List of column names to extract attribute values from the row.

        time_start_property:
            Optional; the column name or datetime representing the start time.

        time_end_property:
            Optional; the column name or datetime representing the end time.

        time_fmt:
            Optional; the format or list of formats for parsing time strings.

    Returns:
        A geospatial object parsed from the feature class row, with attributes and time interval.

    Raises:
        ValueError: If the geometry type is not supported.
    """
    geometry = row.SHAPE
    geometry_type_str = type(geometry).__name__.upper()

    if geometry_type_str not in _PARSER_MAP:
        raise ValueError(f'Unsupported geometry type: {geometry_type_str}.')

    parser = _PARSER_MAP[geometry_type_str]
    properties = {col: getattr(row, col) for col in columns}
    time_start_value, time_end_value = None, None
    if time_start_property is not None:
        time_start_value = getattr(row, time_start_property, None)

    if time_end_property is not None:
        time_end_value = getattr(row, time_end_property, None)

    dt = None
    if time_start_value and isinstance(time_start_value, str):
        dt = TimeInterval.from_str(time_start_value, time_end_value, time_fmt)

    elif time_start_value:
        dt = TimeInterval(time_start_value, time_end_value)

    return parser.from_featureclass(
        geometry,
        dt=dt,
        properties=properties
    )


def parse_arcpy_featureclass(
    row,
    fields,
    time_start_property: Optional[Union[str, datetime]] = None,
    time_end_property: Optional[Union[str, datetime]] = None,
    time_fmt: Optional[Union[str, List[str]]] = None,
):
    """
    Parses an ArcGIS feature class row into a geospatial structure.

    Args:
        row: The row from an ArcGIS feature class, typically obtained using an arcpy cursor.
        fields: List of column names to extract attribute values from the row.
        time_start_property: Optional; the column name or datetime representing the start time.
        time_end_property: Optional; the column name or datetime representing the end time.
        time_fmt: Optional; the format or list of formats for parsing time strings.

    Returns:
        A geospatial object parsed from the feature class row, with attributes and time interval.

    Raises:
        ValueError: If the geometry type is not supported.
    """
    geometry_index = fields.index('SHAPE@')
    time_start_index, time_end_index = None, None
  
    if time_start_property in fields:
        time_start_index = fields.index(time_start_property)

    if time_end_property in fields:
        time_end_index = fields.index(time_end_property)

    geometry = row.SHAPE
    geometry_type_str = type(geometry).__name__.upper()

    if geometry_type_str not in _PARSER_MAP:
        raise ValueError(f'Unsupported geometry type: {geometry_type_str}.')

    parser = _PARSER_MAP[geometry_type_str]

    properties = dict(zip(columns, row))
    del properties['SHAPE@']
    time_start_value, time_end_value = None, None

    if time_start_value is not None:
        time_start_value = getattr(row, time_start_property, None)

    if time_end_value is not None:
        time_end_value = getattr(row, time_end_property, None)

    dt = None
    if time_start_value and isinstance(time_start_value, str):
        dt = TimeInterval.from_str(time_start_value, time_end_value, time_fmt)

    elif time_start_value:
        dt = TimeInterval(time_start_value, time_end_value)

    return parser.from_featureclass(
        geometry,
        dt=dt,
        properties=properties
    )


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
        parser = _PARSER_MAP[kml.geometry.__geo_interface__['type'].upper()]  # type: ignore
        shape = parser.from_fastkml_placemark(kml)
        shape._properties.update(_props)
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
    wkt_type = re.split(r'\s?\(', wkt, 1)[0].upper()
    if wkt_type not in _PARSER_MAP:
        raise ValueError('Invalid WKT.')

    return _PARSER_MAP[wkt_type].from_wkt(wkt)  # type: ignore

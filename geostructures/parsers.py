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
    sedf,
    time_start_property: Optional[str] = None,
    time_end_property: Optional[str] = None,
    _shapes: Optional[List[GeoShape]] = None,
    _props: Optional[Dict[str, str]] = None,
):
    """
    Parses a Spatially Enabled DataFrame (SEDF) from ArcGIS and converts it into geostructures.

    Args:
        sedf (DataFrame):
            A Spatially Enabled DataFrame containing feature class data.

        time_start_property (str, optional):
            The name of the field containing the start time data.

        time_end_property (str, optional):
            The name of the field containing the end time data.

        _shapes (List[GeoShape], optional):
            Internal use only. Mutated with geostructures as they're extracted from the feature class.

        _props (Dict[str, str], optional):
            Internal use only. Information about higher-level containers
            to store as properties on the shape for traceability.

    Returns:
        List[GeoShape]: A list of GeoShape objects parsed from the feature class.
    """
    if _shapes is None:
        _shapes = []
    if _props is None:
        _props = {}

    property_columns = [col for col in sedf.columns if col != 'SHAPE']

    for row in sedf.itertuples():
        geometry = row.SHAPE
        geometry_type_str = type(geometry).__name__.upper()
        if geometry_type_str not in _PARSER_MAP:
            raise ValueError(f"Unsupported geometry type: {geometry_type_str}")

        parser = _PARSER_MAP[geometry_type_str]
        properties = {col: getattr(row, col) for col in property_columns}

        if geometry_type_str == 'POINT':
            lon, lat = geometry.x, geometry.y
            shape = parser(Coordinate(lon, lat), properties=properties)

        elif geometry_type_str == 'MULTIPOINT':
            # Handle multipoint geometry
            points = [Coordinate(point[0], point[1]) for point in geometry['points']]
            shape = parser(points, properties=properties)

        elif geometry_type_str == 'POLYLINE':
            # Extract parts of the polyline and convert to list of Coordinates
            lines = []
            for part in geometry['paths']:
                line = [Coordinate(point[0], point[1]) for point in part]
                lines.append(line)

            # Create the GeoLineString using the entire line
            if len(lines) == 1:
                shape = parser(lines[0], properties=properties)
            else:
                shape = MultiGeoLineString(lines, properties=properties)

        elif geometry_type_str == 'POLYGON':
            # Handle MultiPolygon and single Polygon geometries
            rings = []
            for ring in geometry['rings']:
                # Convert each point in the ring to a Coordinate object
                outline = [Coordinate(point[0], point[1]) for point in ring]
                # Ensure the outline is closed (first point equals the last point)
                if outline[0] != outline[-1]:
                    outline.append(outline[0])
                rings.append(outline)

            if len(rings) == 1:
                # Create a GeoPolygon if there's only one part
                outline = rings[0]
                holes = rings[1:] if len(rings) > 1 else None
                shape = parser(outline, holes=holes, properties=properties)
            else:
                # Create a MultiGeoPolygon if there are multiple parts
                shape = MultiGeoPolygon([parser(ring) for ring in rings], properties=properties)

        else:
            raise TypeError(f'Parser for {geometry_type_str} is not available at this time.')

        # Assign the 'dt' attribute if time fields are provided
        start_time = getattr(row, time_start_property, None) if isinstance(time_start_property, str) else None
        end_time = getattr(row, time_end_property, None) if isinstance(time_end_property, str) else None
        shape.dt = _get_datetime_pandas(start_time, end_time)
        shape._properties.update(_props)
        _shapes.append(shape)

    return _shapes


def parse_arcpy_featureclass(
    cursor,
    fields: Optional[List[str]] = None,
    time_start_property: Optional[str] = None,
    time_end_property: Optional[str] = None,
    _shapes: Optional[List[GeoShape]] = None,
    _props: Optional[Dict[str, str]] = None,
):
    """
    Parses feature class data using arcpy and converts it into geostructures.

    Args:
        cursor (arcpy.da.SearchCursor):
            An arcpy.da.SearchCursor containing the feature class data.

        fields (List[str]):
            List of field names to retrieve from the cursor.

        time_start_property (str, optional):
            The name of the field containing the start time data.

        time_end_property (str, optional):
            The name of the field containing the end time data.

        _shapes (List[GeoShape], optional):
            Internal use only. Mutated with geostructures as they're extracted from the feature class.

        _props (Dict[str, str], optional):
            Internal use only. Information about higher-level containers
            to store as properties on the shape for traceability.

    Returns:
        List[GeoShape]: A list of GeoShape objects parsed from the feature class.
    """
    if _shapes is None:
        _shapes = []
    if _props is None:
        _props = {}

    geometry_index = fields.index('SHAPE@')
    time_start_index = fields.index(time_start_property) if time_start_property in fields else None
    time_end_index = fields.index(time_end_property) if time_end_property in fields else None
    cursor.reset()

    for row in cursor:
        geometry = row[geometry_index]
        geometry_type_str = type(geometry).__name__.upper()

        if geometry_type_str not in _PARSER_MAP:
            raise ValueError(f"Unsupported geometry type: {geometry_type_str}")

        parser = _PARSER_MAP[geometry_type_str]
        properties = dict(zip(fields, row))
        del properties['SHAPE@']

        if geometry_type_str == 'POINT':
            lon, lat = geometry.X, geometry.Y
            shape = parser(Coordinate(lon, lat), properties=properties)

        elif geometry_type_str == 'MULTIPOINT':
            # Handle multipoint geometry
            points = [Coordinate(point.X, point.Y) for point in geometry]
            shape = parser(points, properties=properties)

        elif geometry_type_str == 'POLYLINE':
            # Parse the polyline geometry from the ESRI format to GeoLineString
            lines = []
            for part in geometry.getPart():
                # Convert each point in the part to a Coordinate object
                line = [Coordinate(point.X, point.Y) for point in part]
                lines.append(line)

            # Create the GeoLineString using the entire line
            if len(lines) == 1:
                shape = parser(lines[0], properties=properties)
            else:
                shape = MultiGeoLineString(lines, properties=properties)

        elif geometry_type_str == 'POLYGON':
            # Parse the polygon geometry from the ESRI format to GeoPolygon or MultiGeoPolygon
            rings = []
            for part in geometry.getPart():
                # Convert each point in the ring to a Coordinate object
                outline = [Coordinate(point.X, point.Y) for point in part]
                # Ensure the outline is closed (first point equals the last point)
                if outline[0] != outline[-1]:
                    outline.append(outline[0])
                rings.append(outline)

            if len(rings) == 1:
                # Create a GeoPolygon if there's only one part
                outline = rings[0]
                holes = rings[1:] if len(rings) > 1 else None
                shape = parser(outline, holes=holes, properties=properties)
            else:
                # Create a MultiGeoPolygon if there are multiple parts
                shape = MultiGeoPolygon([parser(ring) for ring in rings], properties=properties)

        else:
            raise TypeError(f'Parser for {geometry_type_str} not available at this time.')

        # Assign the 'dt' attribute if time_start_property or time_end_property is provided and exists in the row
        start_time = row[time_start_index] if isinstance(time_start_property, str) and time_start_index is not None else None
        end_time = row[time_end_index] if isinstance(time_end_property, str) and time_end_index is not None else None
        shape.dt = _get_datetime_pandas(start_time, end_time)
        shape._properties.update(_props)
        _shapes.append(shape)

    return _shapes


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


from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objects import Figure

from geostructures import Coordinate, GeoLineString, GeoPoint, GeoPolygon
from geostructures.structures import GeoShape
from geostructures.collections import FeatureCollection, ShapeCollection
from geostructures.geohash import H3Hasher


def _draw_points(
    points: List[GeoPoint],
    hover_data: Optional[List] = None,
    opacity: Optional[float] = None,
):
    """
    Plots a series of GeoPoints to a plotly graph objects Figure

    Args:
        points:
            A list of GeoPoints

        hover_data:
            A list of properties that should be displayed on shape tooltips

        opacity:
            The desired shape opacity

    Returns:
        go.Figure
    """
    hover_data = hover_data or []

    return px.scatter_mapbox(
        [
            {
                'id': idx,
                'weight': 1.0,
                'lat': shape.centroid.to_float()[1],
                'lon': shape.centroid.to_float()[0],
                'lat/lon': ', '.join(shape.centroid.to_str()),
                **{key: shape.properties.get(key, '') for key in hover_data}
            } for idx, shape in enumerate(points)
        ],
        lon='lon',
        lat='lat',
        color='weight',
        mapbox_style="carto-positron",
        opacity=opacity or 0.5,
        hover_data={
            'lat': False,
            'lon': False,
            **{k: True for k in ('id', 'weight', 'lat/lon', *hover_data)}
        }
    )


def _draw_lines(
    lines: List[GeoLineString],
    hover_data: Optional[List] = None,
):
    """
    Plots a series of GeoLineStrings to a plotly graph objects Figure

    Args:
        lines:
             A list of GeoLineStrings

        hover_data:
            A list of properties that should be displayed on shape tooltips

    Returns:
        go.Figure
    """
    hover_data = hover_data or []

    return px.line_mapbox(
        [
            {
                'id': idx,
                'weight': 1.0,
                'lat': point.to_float()[1],
                'lon': point.to_float()[0],
                'lat/lon': ', '.join(point.to_str()),
                **{key: shape.properties.get(key, '') for key in hover_data}
            } for idx, shape in enumerate(lines) for point in shape.bounding_coords()
        ],
        lon='lon',
        lat='lat',
        color='id',
        mapbox_style="carto-positron",
        hover_data={
            'lat': False,
            'lon': False,
            **{k: True for k in ('id', 'weight', 'lat/lon', *hover_data)}
        },
    )


def _draw_shapes(
    shapes: List[GeoShape],
    hover_data: Optional[List] = None,
    opacity: Optional[float] = None,
):
    """
    Plots a series of GeoShapes (excluding GeoPoints and GeoLineStrings) to a plotly
    graph objects Figure

    Args:
        shapes:
            A list of GeoShapes (cannot be GeoPoints or GeoLineStrings)

        hover_data:
            A list of properties that should be displayed on shape tooltips

        opacity:
            The desired shape opacity

    Returns:
        go.Figure
    """
    hover_data = hover_data or []
    return px.choropleth_mapbox(
        [
            {
                'id': idx,
                'weight': 1.0,
                **{key: shape.properties.get(key, '') for key in hover_data}
            } for idx, shape in enumerate(shapes)
        ],
        geojson=FeatureCollection(shapes).to_geojson(),
        locations='id',
        color='weight',
        mapbox_style="carto-positron",
        opacity=opacity or 0.5,
        featureidkey='id',
        hover_data=['id', *sorted(['weight', *hover_data])],
    )


def draw_collection(
    collection: FeatureCollection,
    hover_data: Optional[List] = None,
    opacity: Optional[float] = None,
):
    """
    Draws a FeatureCollection as a plotly choropleth.

    Shapes will be colored according to the 'weight' property, if present. Otherwise
    all shapes will be given the default weight of 1.0.

    Args:
        collection:
            A geostructures FeatureCollection

        hover_data:
            A list of properties that should be displayed on shape tooltips

        opacity:
            The desired shape opacity

    Returns:
        A plotly figure
    """
    if not hover_data:
        hover_data = sorted(
            set(key for geoshape in collection.geoshapes for key in geoshape.properties)
        )

    avg_lon, avg_lat = [
        np.average(x)
        for x in list(zip(*[geoshape.centroid.to_float() for geoshape in collection.geoshapes]))
    ]

    _points = [x for x in collection if isinstance(x, GeoPoint)]
    _lines = [x for x in collection if isinstance(x, GeoLineString)]
    _shapes = [x for x in collection if not isinstance(x, (GeoLineString, GeoPoint))]

    _fig = go.Figure()
    if _points:
        _fig.add_trace(_draw_points(_points, hover_data, opacity).data[0])
    if _lines:
        for trace in _draw_lines(_lines, hover_data).data:
            _fig.add_trace(trace)
    if _shapes:
        _fig.add_trace(_draw_shapes(_shapes, hover_data, opacity).data[0])

    _fig.update_geos(
        fitbounds='locations',
    )
    _fig.update_layout(
        coloraxis_showscale=False,  # legends will overlap if not removed
        showlegend=False,
        mapbox_style="carto-positron",
        mapbox_center={'lat': avg_lat, 'lon': avg_lon},
    )
    return _fig


def h3_choropleth(
        hexmap: Dict[str, float],
        min_weight: float = 0,
        fig: Figure = None,
        property_map: Optional[Dict[str, Any]] = None,
):
    """
    Helper for quickly plotting h3 hashes on a map.

    Args:
        hexmap:
            A dictionary of h3 hexagon ids to their corresponding weights.

        min_weight (float):
            The minimum weight a hex must have in order to be displayed.

        fig (plotly.graph_objs.Figure): (Default None)
            An existing plotly figure. If provided, the created visualization will be inserted
            into this figure instead of returning a brand new one.

        property_map:
            A dictionary of additional properties to include. Keys should be a given hex id, and
            values should be a dictionary of the properties. All values must contain the same
            keys.

    Returns:
        (plotly.graph_objs.Figure)
    """
    import h3

    property_map = property_map or {}
    prop_keys = set(key for hex_id, props in property_map.items() for key in props)

    if min_weight:
        hexmap = {k: v for k, v in hexmap.items() if v > min_weight}

    polygons = []
    _lats, _lons = [], []
    for _hex, weight in hexmap.items():
        boundary = h3.h3_to_geo_boundary(_hex)
        poly = GeoPolygon(
            [Coordinate(*x[::-1]) for x in [*boundary, boundary[0]]],
            properties={'weight': weight},
        )
        polygons.append(poly.to_geojson(id=_hex, properties=property_map))
        _lat, _lon = h3.h3_to_geo(_hex)
        _lats.append(_lat)
        _lons.append(_lon)

    avg_lat, avg_lon = np.average(_lats), np.average(_lons)
    gjson = {
        'type': 'FeatureCollection',
        'features': polygons,
    }

    _fig = px.choropleth_mapbox(
        [
            {
                'id': hex_id,
                'weight': weight,
                **{
                    key: property_map.get(hex_id, {}).get(key, '')
                    for key in prop_keys
                },
            } for hex_id, weight in hexmap.items()
        ],
        geojson=gjson,
        locations='id',
        color='weight',
        mapbox_style="carto-positron",
        opacity=0.5,
        featureidkey='id',
        hover_data=['id', 'weight', *prop_keys],
    )
    _fig.update_layout(
        mapbox_center={'lat': avg_lat, 'lon': avg_lon},
    )

    if fig:
        fig.add_trace(_fig.data[0])
        return fig

    _fig.update_geos(fitbounds='locations')

    return _fig


def collectionmap_h3_choropleth(
    collection_map: Dict[str, ShapeCollection],
    hasher: H3Hasher,
    min_weight: float = 0,
    fig: Figure = None
):
    """
    Helper for quickly plotting h3 hashes on a map.

    Args:
        collection_map:
            A dictionary of unique ids to their corresponding shape collections.

        hasher:
            A H3Hasher object, provided by the geostructures.geohash module.

        min_weight:
            The minimum weight a hex must have (cumulative between tracks) in order to be displayed.

        fig: (Default None)
            An existing plotly figure. If provided, the created visualization will be inserted
            into this figure instead of returning a brand new one.

    Returns:
        (plotly.graph_objs.Figure)
    """
    display_hexes = defaultdict(lambda: 0)
    hexid_to_trackid_map = defaultdict(set)
    for trackid, collection in collection_map.items():
        for shape in collection.geoshapes:
            hexes = hasher.hash_shape(shape)
            for hex_id in hexes:
                display_hexes[hex_id] += 1
                hexid_to_trackid_map[hex_id].add(str(trackid))

    property_map = {
        hex_id: {'collections': ', '.join(hexid_to_trackid_map[hex_id])}
        for hex_id in display_hexes.keys()
    }
    return h3_choropleth(
        display_hexes,
        min_weight=min_weight,
        property_map=property_map,
        fig=fig
    )

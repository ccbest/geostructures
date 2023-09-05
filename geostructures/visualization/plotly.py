
from typing import List, Optional

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from geostructures.structures import GeoLineString, GeoPoint, GeoShape
from geostructures.collections import FeatureCollection


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
                'lat': shape.centroid.to_float()[1],
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

    avg_lat, avg_lon = [
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
        center={'lat': avg_lat, 'lon': avg_lon},
    )
    _fig.update_layout(
        coloraxis_showscale=False,  # legends will overlap if not removed
        showlegend=False,
        mapbox_style="carto-positron",
    )
    return _fig

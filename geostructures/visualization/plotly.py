
from typing import List, Optional

import numpy as np
import plotly.express as px

from geostructures.collections import FeatureCollection


def draw_collection(
    collection: FeatureCollection,
    hover_data: Optional[List] = None,
    opacity: Optional[float] = None,
):  # pragma: no cover
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

    _fig = px.choropleth_mapbox(
        [
            {
                'id': idx,
                'weight': 1.0,
                **{key: shape.properties.get(key, '') for key in hover_data}
            } for idx, shape in enumerate(collection.geoshapes)
        ],
        geojson=collection.to_geojson(),
        locations='id',
        color='weight',
        mapbox_style="carto-positron",
        opacity=opacity or 0.5,
        featureidkey='id',
        hover_data=['id', *sorted(['weight', *hover_data])],
        center={'lat': avg_lat, 'lon': avg_lon}
    )

    _fig.update_geos(fitbounds='locations')

    return _fig

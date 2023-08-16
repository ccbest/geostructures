
import plotly.express as px

from geostructures.collections import Collection


def draw_collection(collection: Collection):
    _fig = px.choropleth_mapbox(
        [
            {
                'id': idx,
                'weight': 1.0,
            } for idx, shape in enumerate(collection.geoshapes)
        ],
        geojson=collection.to_geojson(),
        locations='id',
        color='weight',
        mapbox_style="carto-positron",
        opacity=0.5,
        featureidkey='id',
        hover_data=['id', 'weight', ],
        center={'lat': 0.0, 'lon': 1.0}
    )

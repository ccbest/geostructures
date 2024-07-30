"""Module for geostructures type hinting"""

__all__ = [
    'GeoShape', 'LineLike', 'MultiShape', 'PointLike',
    'PolygonLike', 'SingleShape', 'SinglePolygon'
]

from typing import Union

from geostructures._base import (
    BaseShape,
    MultiShapeBase,
    LineLikeMixin,
    PointLikeMixin,
    PolygonLikeMixin,
)
from geostructures.structures import PolygonBase, GeoLineString, GeoPoint

# Any Shape
GeoShape = BaseShape

# Individual and Multi-Shapes
SingleShape = Union[PolygonBase, GeoLineString, GeoPoint]
MultiShape = MultiShapeBase

# All Single Polygon Shapes, e.g. GeoBox
SinglePolygon = PolygonBase

# isinstance with type aliases only supported in Python 3.10+, have to use mixins
LineLike = LineLikeMixin  # Union[GeoLineString, MultiGeoLinestring]
PointLike = PointLikeMixin  # Union[GeoPoint, MultiGeoPoint]
PolygonLike = PolygonLikeMixin  # Union[PolygonBase, MultiGeoPolygon]

"""Module for geostructures type hinting"""

__all__ = [
    'GeoShape', 'LineLike', 'MultiShape', 'PointLike',
    'PolygonLike', 'SingleShape', 'SinglePolygon',
    'SimpleShape', 'ShapeCollection',
]

from geostructures._base import (
    BaseShape,
    MultiShapeBase,
    SingleShapeBase,
    SimpleShapeMixin,
    LineLikeMixin,
    PointLikeMixin,
    PolygonLikeMixin,
)
from geostructures.collections import CollectionBase
from geostructures.structures import PolygonBase

# Any Shape
GeoShape = BaseShape

# Individual and Multi-Shapes
SingleShape = SingleShapeBase  # Union[PolygonBase, GeoLineString, GeoPoint]
MultiShape = MultiShapeBase  # Union[MultiGeoPolygon, MultiGeoLineString, MultiGeoPoint]

SimpleShape = type[SimpleShapeMixin]

# All Single Polygon Shapes, e.g. GeoBox
SinglePolygon = PolygonBase  # Union[GeoPolygon, GeoBox, GeoCircle, ... ]

# isinstance with type aliases only supported in Python 3.10+, have to use mixins
LineLike = LineLikeMixin  # Union[GeoLineString, MultiGeoLinestring]
PointLike = PointLikeMixin  # Union[GeoPoint, MultiGeoPoint]
PolygonLike = PolygonLikeMixin  # Union[PolygonBase, MultiGeoPolygon]

ShapeCollection = type[CollectionBase]

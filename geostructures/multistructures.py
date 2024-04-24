
import copy
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from geostructures._base import _GEOTIME_TYPE, _SHAPE_TYPE
from geostructures.calc import haversine_distance_meters
from geostructures.coordinates import Coordinate
from geostructures.structures import BaseShape, GeoCircle, GeoLineString, GeoPoint, GeoPolygon


class BaseMultiGeoShape(BaseShape):

    pass


class MultiGeoPolygon(BaseMultiGeoShape):

    def __init__(
        self,
        polygons: List[GeoPolygon],
        dt: Optional[_GEOTIME_TYPE] = None,
        properties: Optional[Dict] = None,
    ):
        super().__init__(dt, properties)
        self.geoshapes = polygons

    def __hash__(self) -> int:
        return hash(tuple(hash(x) for x in self.geoshapes))

    def __repr__(self):
        return f'<MultiGeoPolygon of {len(self.geoshapes)} shapes>'

    def area(self) -> float:
        return sum(x.area for x in self.geoshapes)

    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        lons, lats = list(zip(x.bounds for x in self.geoshapes))
        return (min(lons), max(lons)), (min(lats), max(lats))

    @cached_property
    def centroid(self):
        # Decompose polygon into triangles using vertex pairs around the origin
        poly1 = np.array([x.to_float() for poly in self.geoshapes for x in poly.bounding_coords()])
        poly2 = np.roll(poly1, -1, axis=0)
        # Find signed area of each triangle
        signed_areas = 0.5 * np.cross(poly1, poly2)
        # Return average of triangle centroids, weighted by area
        return Coordinate(*np.average((poly1 + poly2) / 3, axis=0, weights=signed_areas))

    def bounding_coords(self, **kwargs) -> List[List[Coordinate]]:
        return [x.bounding_coords(**kwargs) for x in self.geoshapes]

    def bounding_edges(self, **kwargs) -> List[List[Tuple[Coordinate, Coordinate]]]:
        return [x.bounding_edges(**kwargs) for x in self.geoshapes]

    def circumscribing_circle(self) -> 'GeoCircle':
        centroid = self.centroid
        max_dist = max(
            haversine_distance_meters(x, centroid)
            for poly in self.geoshapes
            for x in poly.bounding_coords()
        )
        return GeoCircle(centroid, max_dist, dt=self.dt)

    def contains_coordinate(self, coord: Coordinate) -> bool:
        for poly in self.geoshapes:
            if poly.contains_coordinate(coord):
                return True

        return False

    def contains_shape(self, shape: 'BaseShape', **kwargs) -> bool:
        for poly in self.geoshapes:
            if poly.contains_shape(shape):
                return True

        return False

    def copy(self: _SHAPE_TYPE) -> _SHAPE_TYPE:
        return MultiGeoPolygon(
            [x.copy() for x in self.polygons],
            dt=self.dt,
            properties=copy.deepcopy(self._properties)
        )

    @classmethod
    def from_geojson(
        cls,
        gjson: Dict[str, Any],
        time_start_property: str = 'datetime_start',
        time_end_property: str = 'datetime_end',
        time_format: Optional[str] = None,
    ):
        # TODO
        pass

    def linear_rings(self, **kwargs) -> List[List[List[Coordinate]]]:
        """
        Produce a list of linear rings for the object, where the first ring is the outermost
        shell and the following rings are holes.

        Using discrete coordinates to represent a continuous curve implies some level of data
        loss. You can minimize this loss by specifying k, which represents the number of
        points drawn.

        All shapes that represent a linear ring (e.g. a box or polygon) will return
        self-closing coordinates, meaning the last coordinate is equal to the first.

        Keyword Args:
            k: (int)
                For shapes with smooth curves, increasing k increases the number of
                points generated along the curve
        """
        return [
            [
                poly.bounding_coords(**kwargs),
                *[list(reversed(hole.bounding_coords())) for hole in poly.holes]
            ] for poly in self.geoshapes
        ]

    def to_geojson(
        self,
        k: Optional[int] = None,
        properties: Optional[Dict] = None,
        **kwargs
    ) -> Dict:
        """
        Convert the shape to geojson format.

        Optional Args:
            k: (int)
                For shapes with smooth curves, defines the number of points
                generated along the curve.

            properties: (dict)
                Any number of properties to be included in the geojson properties. These
                values will be unioned with the shape's already defined properties (and
                override them where keys conflict)

        Returns:
            (dict)
        """

    def _to_shapely(self, **kwargs):
        """
        Converts the geoshape into a Shapely shape.
        """
        import shapely  # pylint: disable=import-outside-toplevel

        converted = []
        for shape in self.linear_rings(**kwargs):
            shell, holes = shape[0], []
            if len(shape) > 1:
                holes = shape[1:]

            converted.append(
                (
                    tuple(coord.to_float() for coord in shell),
                    [
                        tuple(coord.to_float() for coord in ring)
                        for ring in holes[1:]
                    ]
                )
            )

        return shapely.geometry.MultiPolygon(converted)

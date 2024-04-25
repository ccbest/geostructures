

__all__ = ['MultiGeoPolygon']

import copy
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from geostructures._base import (
    _GEOTIME_TYPE, _RE_MULTIPOLYGON_WKT, _RE_LINEAR_RING, _RE_LINEAR_RINGS, _SHAPE_TYPE,
    BaseShape, get_dt_from_geojson_props, parse_wkt_linear_ring
)
from geostructures.calc import haversine_distance_meters
from geostructures.coordinates import Coordinate
from geostructures.structures import GeoCircle, GeoLineString, GeoPoint, GeoPolygon


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
        pl = "s" if len(self.geoshapes) != 1 else ""
        return f'<MultiGeoPolygon of {len(self.geoshapes)} polygon{pl}>'

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
        geom = gjson.get('geometry', {})
        if not geom.get('type') == 'MultiPolygon':
            raise ValueError(
                f'Geometry represents a {geom.get("type")}; expected MultiPolygon.'
            )

        shapes = []
        for shape in geom.get('coordinates', []):
            rings = [[Coordinate(x, y) for x, y in ring] for ring in shape]
            shell, holes = rings[0], []
            if len(rings) > 1:
                holes = [GeoPolygon(list(reversed(ring))) for ring in rings[1:]]

            shapes.append(GeoPolygon(shell, holes=holes))

        properties = gjson.get('properties', {})
        dt = get_dt_from_geojson_props(
            properties,
            time_start_property,
            time_end_property,
            time_format
        )
        return MultiGeoPolygon(
            shapes,
            dt=dt,
            properties=properties
        )

    @classmethod
    def from_shapely(
        cls,
        multipolygon
    ):
        """
        Creates a GeoPolygon from a shapely polygon

        Args:
            multipolygon:
                A shapely multipolygon

        Returns:
            GeoPolygon
        """
        return cls.from_wkt(multipolygon.wkt)

    @classmethod
    def from_wkt(
        cls,
        wkt_str: str,
        dt: Optional[_GEOTIME_TYPE] = None,
        properties: Optional[Dict] = None
    ) -> 'MultiGeoPolygon':
        """Create a GeoPolygon from a wkt string"""
        if not _RE_MULTIPOLYGON_WKT.match(wkt_str):
            raise ValueError(f'Invalid WKT Polygon: {wkt_str}')

        shapes = []
        for shape in _RE_LINEAR_RINGS.findall(wkt_str):
            coord_groups = _RE_LINEAR_RING.findall(shape)
            shell, holes = parse_wkt_linear_ring(coord_groups[0]), []

            if len(coord_groups) > 1:
                holes = [
                    GeoPolygon(list(reversed(parse_wkt_linear_ring(coord_group))))
                    for coord_group in coord_groups[1:]
                ]

            shapes.append(GeoPolygon(shell, holes=holes))

        return MultiGeoPolygon(
            shapes,
            dt=dt,
            properties=properties
        )

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
        return {
            'type': 'Feature',
            'geometry': {
                'type': 'MultiPolygon',
                'coordinates': [
                    [
                        [
                            list(coord.to_float()) for coord in ring
                        ] for ring in shape
                    ]
                    for shape in self.linear_rings(k=k)
                ]
            },
            'properties': {
                **self.properties,
                **self._dt_to_json(),
                **(properties or {})
            },
            **kwargs
        }

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
                        for ring in holes
                    ]
                )
            )

        return shapely.geometry.MultiPolygon(converted)

    def to_wkt(self, **kwargs) -> str:
        """
        Converts the shape to its WKT string representation

        Keyword Args:
            Arguments to be passed to the .bounding_coords() method. Reference
            that method for a list of corresponding kwargs.

        Returns:
            str
        """
        bbox_strs = []
        for shape in self.linear_rings(**kwargs):
            bbox_strs.append('(' + ', '.join(
                [self._linear_ring_to_wkt(ring) for ring in shape]
            ) + ')')

        return f'MULTIPOLYGON({", ".join(bbox_strs)})'



__all__ = ['MultiGeoShape', 'MultiGeoLineString', 'MultiGeoPoint']


import copy
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple, Sequence

import numpy as np

from geostructures._base import (
    _RE_MULTIPOLYGON_WKT, _RE_MULTIPOINT_WKT,
    _RE_MULTILINESTRING_WKT, _RE_LINEAR_RING, _RE_LINEAR_RINGS,
    ShapeLike, MultiShapeBase, parse_wkt_linear_ring,
    PointLike, LineLike
)
from geostructures.time import GEOTIME_TYPE
from geostructures._geometry import convex_hull
from geostructures.calc import haversine_distance_meters
from geostructures.coordinates import Coordinate
from geostructures.structures import GeoCircle, GeoLineString, GeoPoint, GeoPolygon, ShapeBase
from geostructures.utils.functions import get_dt_from_geojson_props
from geostructures.utils.logging import warn_once


class MultiGeoLineString(MultiShapeBase, LineLike):

    def __init__(
        self,
        geoshapes: List[GeoLineString],
        dt: Optional[GEOTIME_TYPE] = None,
        properties: Optional[Dict] = None,
    ):
        super().__init__(dt, properties)
        self.geoshapes: List[GeoLineString] = geoshapes

    def __repr__(self):
        pl = "s" if len(self.geoshapes) != 1 else ""
        return f'<MultiGeoLineString of {len(self.geoshapes)} linestring{pl}>'

    @cached_property
    def centroid(self):
        # TODO: weighted by line length
        lon, lat = np.mean(
            np.array([coord.to_float() for shape in self.geoshapes for coord in shape.vertices]),
            axis=0
        )
        return Coordinate(lon, lat)

    @property
    def segments(self) -> List[List[Tuple[Coordinate, Coordinate]]]:
        return [x.segments for x in self.geoshapes]

    def circumscribing_circle(self) -> 'GeoCircle':
        centroid = self.centroid
        max_dist = max(
            haversine_distance_meters(coord, centroid)
            for shape in self.geoshapes
            for coord in shape.vertices
        )
        return GeoCircle(centroid, max_dist, dt=self.dt)

    def convex_hull(self) -> GeoPolygon:
        return GeoPolygon(
            convex_hull([coord for shape in self.geoshapes for coord in shape.vertices])
        )

    def copy(self) -> 'MultiGeoLineString':
        return MultiGeoLineString(
            [x.copy() for x in self.geoshapes],
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
    ) -> 'MultiGeoLineString':
        geom = gjson.get('geometry', {})
        if not geom.get('type') == 'MultiLineString':
            raise ValueError(
                f'Geometry represents a {geom.get("type")}; expected MultiLineString.'
            )

        lines = [
            GeoLineString([Coordinate(*coord) for coord in line])
            for line in geom.get('coordinates', [])
        ]
        properties = gjson.get('properties', {})
        dt = get_dt_from_geojson_props(
            properties,
            time_start_property,
            time_end_property,
            time_format
        )
        return MultiGeoLineString(
            lines,
            dt=dt,
            properties=properties
        )

    @classmethod
    def from_pyshp(cls, shape, dt: Optional[GEOTIME_TYPE] = None, properties: Optional[Dict] = None):
        """
        Create a MultiGeoLineString from a pyshyp polyline.

        Args:
            shape:
                A polygon from the pyshp library

            dt:
                Optional time bounds (presented as a datetime or geostructures.TimeInterval)

            properties:
                Optional additional properties to associate to the shape

        Returns:
            MultiGeoLineString
        """
        properties = properties or {}
        if hasattr(shape, 'z'):  # pragma: no cover
            properties['Z'] = shape.z
            warn_once(
                'Shapefile contains unsupported Z data; Z-values will be '
                'stored in shape properties'
            )

        if hasattr(shape, 'm'):  # pragma: no cover
            properties['M'] = shape.m
            warn_once(
                'Shapefile contains unsupported M data; M-values will be '
                'stored in shape properties'
            )

        mgls = MultiGeoLineString.from_geojson(
            {
                'type': 'Feature',
                'geometry': shape.__geo_interface__,
                'properties': properties
            }
        )
        mgls.set_dt(dt)
        return mgls

    @classmethod
    def from_shapely(
        cls,
        shape
    ):
        """
        Creates a corresponding multi shape from a shapely object

        Args:
            shape:
                A shapely object

        Returns:
            MultiGeoLineString
        """
        return cls.from_wkt(shape.wkt)

    @classmethod
    def from_wkt(
        cls,
        wkt_str: str,
        dt: Optional[GEOTIME_TYPE] = None,
        properties: Optional[Dict] = None
    ) -> 'MultiGeoLineString':
        """Create a GeoMultiLineString from a wkt string"""
        if not _RE_MULTILINESTRING_WKT.match(wkt_str):
            raise ValueError(f'Invalid WKT MultiLineString: {wkt_str}')

        lines = [
            GeoLineString(parse_wkt_linear_ring(line))
            for line in _RE_LINEAR_RING.findall(wkt_str)
        ]

        return MultiGeoLineString(
            lines,
            dt=dt,
            properties=properties
        )

    def to_geojson(
        self,
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
                'type': 'MultiLineString',
                'coordinates': [
                    [
                        list(vertex.to_float()) for vertex in shape.vertices
                    ]
                    for shape in self.geoshapes
                ]
            },
            'properties': {
                **self._properties_json,
                **(properties or {})
            },
            **kwargs
        }

    def to_pyshp(self, writer):
        return writer.line(
            [[list(vertex.to_float()) for vertex in shape.vertices] for shape in self.geoshapes]
        )

    def _to_shapely(self, **kwargs):
        """
        Converts the geoshape into a Shapely shape.
        """
        import shapely  # pylint: disable=import-outside-toplevel

        lines = [
            [x.to_float() for x in shape.vertices]
            for shape in self.geoshapes
        ]

        return shapely.geometry.MultiLineString(lines)

    def to_wkt(self, **_) -> str:
        """
        Converts the shape to its WKT string representation

        Keyword Args:
            Arguments to be passed to the .bounding_coords() method. Reference
            that method for a list of corresponding kwargs.

        Returns:
            str
        """
        lines = [self._linear_ring_to_wkt(shape.vertices) for shape in self.geoshapes]
        return f'MULTILINESTRING({", ".join(lines)})'


class MultiGeoPoint(MultiShapeBase, PointLike):

    def __init__(
        self,
        geoshapes: List[GeoPoint],
        dt: Optional[GEOTIME_TYPE] = None,
        properties: Optional[Dict] = None,
    ):
        super().__init__(dt, properties)
        self.geoshapes: List[GeoPoint] = geoshapes

    def __hash__(self) -> int:
        return hash(tuple(hash(x) for x in self.geoshapes))

    def __repr__(self):
        pl = "s" if len(self.geoshapes) != 1 else ""
        return f'<MultiGeoPoint of {len(self.geoshapes)} point{pl}>'

    @cached_property
    def centroid(self):
        return Coordinate(*np.average(
            np.array([point.centroid.to_float() for point in self.geoshapes]),
            axis=0
        ))

    def circumscribing_circle(self) -> 'GeoCircle':
        centroid = self.centroid
        max_dist = max(
            haversine_distance_meters(point.centroid, centroid)
            for point in self.geoshapes
        )
        return GeoCircle(centroid, max_dist, dt=self.dt)

    def convex_hull(self, **_) -> GeoPolygon:
        return GeoPolygon(
            convex_hull([shape.centroid for shape in self.geoshapes]),
        )

    def copy(self) -> 'MultiGeoPoint':
        return MultiGeoPoint(
            [x.copy() for x in self.geoshapes],
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
    ) -> 'MultiGeoPoint':
        geom = gjson.get('geometry', {})
        if not geom.get('type') == 'MultiPoint':
            raise ValueError(
                f'Geometry represents a {geom.get("type")}; expected MultiPoint.'
            )

        points = [GeoPoint(Coordinate(*coord)) for coord in geom.get('coordinates', [])]
        properties = gjson.get('properties', {})
        dt = get_dt_from_geojson_props(
            properties,
            time_start_property,
            time_end_property,
            time_format
        )
        return MultiGeoPoint(
            points,
            dt=dt,
            properties=properties
        )

    @classmethod
    def from_pyshp(cls, shape, dt: Optional[GEOTIME_TYPE] = None, properties: Optional[Dict] = None):
        """
        Create a MultiGeoPoint from a pyshyp multipoint.

        Args:
            shape:
                A multipoint from the pyshp library

            dt:
                Optional time bounds (presented as a datetime or geostructures.TimeInterval)

            properties:
                Optional additional properties to associate to the shape

        Returns:
            MultiGeoPoint
        """
        properties = properties or {}
        if hasattr(shape, 'z'):  # pragma: no cover
            properties['Z'] = shape.z
            warn_once(
                'Shapefile contains unsupported Z data; Z-values will be '
                'stored in shape properties'
            )

        if hasattr(shape, 'm'):  # pragma: no cover
            properties['M'] = shape.m
            warn_once(
                'Shapefile contains unsupported M data; M-values will be '
                'stored in shape properties'
            )

        mgp = MultiGeoPoint.from_geojson(
            {
                'type': 'Feature',
                'geometry': shape.__geo_interface__,
                'properties': properties
            }
        )
        mgp.set_dt(dt)
        return mgp

    @classmethod
    def from_shapely(
        cls,
        multipoint
    ):
        """
        Creates a GeoPolygon from a shapely polygon

        Args:
            multipoint:
                A shapely multipoint

        Returns:
            GeoPolygon
        """
        return cls.from_wkt(multipoint.wkt)

    @classmethod
    def from_wkt(
        cls,
        wkt_str: str,
        dt: Optional[GEOTIME_TYPE] = None,
        properties: Optional[Dict] = None
    ) -> 'MultiGeoPoint':
        """Create a GeoPolygon from a wkt string"""
        if not _RE_MULTIPOINT_WKT.match(wkt_str):
            raise ValueError(f'Invalid WKT MultiPoint: {wkt_str}')

        points = [
            GeoPoint(coord) for coord in parse_wkt_linear_ring(_RE_LINEAR_RING.findall(wkt_str)[0])
        ]

        return MultiGeoPoint(
            points,
            dt=dt,
            properties=properties
        )

    def to_geojson(
        self,
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
                'type': 'MultiPoint',
                'coordinates': [
                    list(point.centroid.to_float())
                    for point in self.geoshapes
                ]
            },
            'properties': {
                **self._properties_json,
                **(properties or {})
            },
            **kwargs
        }

    def to_pyshp(self, writer):
        return writer.multipoint([x.centroid.to_float() for x in self.geoshapes])

    def _to_shapely(self, **kwargs):
        """
        Converts the geoshape into a Shapely shape.
        """
        import shapely  # pylint: disable=import-outside-toplevel

        points = [x.centroid.to_float() for x in self.geoshapes]

        return shapely.geometry.MultiPoint(points)

    def to_wkt(self, **_) -> str:
        """
        Converts the shape to its WKT string representation

        Keyword Args:
            Arguments to be passed to the .bounding_coords() method. Reference
            that method for a list of corresponding kwargs.

        Returns:
            str
        """
        points = [' '.join(x.centroid.to_str()) for x in self.geoshapes]
        return f'MULTIPOINT({", ".join(points)})'


class MultiGeoShape(MultiShapeBase, ShapeLike):

    def __init__(
        self,
        geoshapes: Sequence[ShapeBase],
        dt: Optional[GEOTIME_TYPE] = None,
        properties: Optional[Dict] = None,
    ):
        super().__init__(dt, properties)
        self.geoshapes = list(geoshapes)

    def __repr__(self):
        pl = "s" if len(self.geoshapes) != 1 else ""
        return f'<MultiGeoShape of {len(self.geoshapes)} shape{pl}>'

    @property
    def area(self) -> float:
        return sum(x.area for x in self.geoshapes)

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

    def convex_hull(self, **kwargs) -> GeoPolygon:
        return GeoPolygon(
            convex_hull([
                coord for shape in self.geoshapes for coord in shape.bounding_coords(**kwargs)
            ])
        )

    def copy(self) -> 'MultiGeoShape':
        return MultiGeoShape(
            [x.copy() for x in self.geoshapes],
            dt=self.dt,
            properties=copy.deepcopy(self._properties)
        )

    def edges(self, **kwargs) -> List[List[List[Tuple[Coordinate, Coordinate]]]]:
        """
        Returns lists of edges, defined as a 2-tuple (start and end) of coordinates, for the
        outer boundary as well as holes in the polygon.

        Operates similar to the `bounding_edges` method but includes information about holes
        in the shape.

        Using discrete coordinates to represent a continuous curve implies some level of data
        loss. You can minimize this loss by specifying k, which represents the number of
        points drawn.

        Does not include information about holes - see the edges method.

        Keyword Args:
            k: (int)
                For shapes with smooth curves, increasing k increases the number of
                points generated along the curve

        Returns:
            Lists of 2-tuples representing edges. The first list will always represent
            the shape's boundary, and any following lists will represent holes.
        """
        return [shape.edges(**kwargs) for shape in self]

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
            shell, holes = rings[0], None
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
        return MultiGeoShape(
            shapes,
            dt=dt,
            properties=properties
        )

    @classmethod
    def from_pyshp(cls, shape, dt: Optional[GEOTIME_TYPE] = None, properties: Optional[Dict] = None):
        """
        Create a MultiGeoShape from a pyshyp multipolygon.

        Args:
            shape:
                A polygon from the pyshp library that contains multiple polygons

            dt:
                Optional time bounds (presented as a datetime or geostructures.TimeInterval)

            properties:
                Optional additional properties to associate to the shape

        Returns:
            MultiGeoShape
        """
        properties = properties or {}
        if hasattr(shape, 'z'):  # pragma: no cover
            properties['Z'] = shape.z
            warn_once(
                'Shapefile contains unsupported Z data; Z-values will be '
                'stored in shape properties'
            )

        if hasattr(shape, 'm'):  # pragma: no cover
            properties['M'] = shape.m
            warn_once(
                'Shapefile contains unsupported M data; M-values will be '
                'stored in shape properties'
            )

        mgp = MultiGeoShape.from_geojson(
            {
                'type': 'Feature',
                'geometry': shape.__geo_interface__,
                'properties': properties
            }
        )
        mgp.set_dt(dt)
        return mgp

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
        dt: Optional[GEOTIME_TYPE] = None,
        properties: Optional[Dict] = None
    ) -> 'MultiGeoShape':
        """Create a GeoPolygon from a wkt string"""
        if not _RE_MULTIPOLYGON_WKT.match(wkt_str):
            raise ValueError(f'Invalid WKT MultiPolygon: {wkt_str}')

        shapes = []
        for shape in _RE_LINEAR_RINGS.findall(wkt_str):
            coord_groups = _RE_LINEAR_RING.findall(shape)
            shell, holes = parse_wkt_linear_ring(coord_groups[0]), None

            if len(coord_groups) > 1:
                holes = [
                    GeoPolygon(list(reversed(parse_wkt_linear_ring(coord_group))))
                    for coord_group in coord_groups[1:]
                ]

            shapes.append(GeoPolygon(shell, holes=holes))

        return MultiGeoShape(
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
            poly.linear_rings(**kwargs) for poly in self.geoshapes
        ]

    def to_geojson(
        self,
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
                    for shape in self.linear_rings(k=kwargs.pop('k', None))
                ]
            },
            'properties': {
                **self._properties_json,
                **(properties or {})
            },
            **kwargs
        }

    def to_pyshp(self, writer):
        return writer.poly(
            [
                # ESRI defines right hand rule as opposite of GeoJSON
                [list(coord.to_float()) for coord in ring[::-1]]
                for shape in self.geoshapes
                for ring in shape.linear_rings()
            ]
        )

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



__all__ = ['MultiGeoPolygon', 'MultiGeoLineString', 'MultiGeoPoint']


import copy
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple, Sequence, cast

import numpy as np

from geostructures._base import (
    _RE_MULTIPOLYGON_WKT, _RE_MULTIPOINT_WKT,
    _RE_MULTILINESTRING_WKT, _RE_LINEAR_RING, _RE_LINEAR_RINGS,
    PolygonLikeMixin, MultiShapeBase,
    PointLikeMixin, LineLikeMixin, SimpleShapeMixin
)
from geostructures.time import GEOTIME_TYPE
from geostructures._geometry import convex_hull
from geostructures.calc import haversine_distance_meters
from geostructures.coordinates import Coordinate
from geostructures.structures import GeoCircle, GeoLineString, GeoPoint, GeoPolygon, PolygonBase
from geostructures.time import TimeInterval
# from geostructures.utils.functions import get_dt_from_geojson_props


class MultiGeoLineString(MultiShapeBase, LineLikeMixin, SimpleShapeMixin):

    def __init__(
        self,
        geoshapes: List[GeoLineString],
        dt: Optional[GEOTIME_TYPE] = None,
        properties: Optional[Dict] = None,
    ):
        super().__init__(dt, properties)
        self.geoshapes: List[GeoLineString] = geoshapes

    @property
    def __geo_interface__(self):
        return {
            'type': 'MultiLineString',
            'coordinates': [
                [
                    list(vertex.to_float()) for vertex in shape.vertices
                ]
                for shape in self.geoshapes
            ]
        }

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
        if 'coordinates' in gjson:
            geom = gjson
        else:
            geom = gjson.get('geometry', {})

        if not geom.get('type') == 'MultiLineString':
            raise ValueError(
                f'Geometry represents a {geom.get("type")}; expected MultiLineString.'
            )

        lines = [
            GeoLineString(
                [Coordinate(**dict(zip(('longitude', 'latitude', 'z'), x))) for x in line]
            ) for line in geom.get('coordinates', [])
        ]
        properties = gjson.get('properties', {})
        dt = None
        if time_start_property in properties or time_end_property in properties:
            # Pop time field so it doesn't remain in properties
            dt_start = properties.pop(time_start_property, None)
            dt_end = properties.pop(time_end_property, None)

            if dt_start and not dt_end:
                dt = TimeInterval.from_str(dt_start, dt_start, time_format)

            elif dt_end and not dt_start:
                dt = TimeInterval.from_str(dt_end, dt_end, time_format)

            else:
                dt = TimeInterval.from_str(dt_start, dt_end, time_format)

        return MultiGeoLineString(
            lines,
            dt=dt,
            properties=properties
        )

    @classmethod
    def from_pyshp(
        cls,
        shape,
        dt: Optional[GEOTIME_TYPE] = None,
        properties: Optional[Dict] = None
    ):
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
        linestrings = []
        z = shape.z if hasattr(shape, 'z') else None
        m = shape.m if hasattr(shape, 'm') else None
        for linestring in shape.__geo_interface__.get('coordinates', []):
            linestrings.append(
                GeoLineString(
                    [
                        Coordinate(
                            *cast(Tuple[float, float], coord),
                            z=z.pop(0) if z else None,
                            m=m.pop(0) if m else None,
                        ) for coord in linestring
                    ]
                )
            )

        return MultiGeoLineString(linestrings, dt=dt, properties=properties)

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

        lines = []
        for linear_ring in _RE_LINEAR_RING.findall(wkt_str):
            coords = cls._parse_wkt_linear_ring(wkt_str, linear_ring)
            lines.append(GeoLineString(coords))

        return MultiGeoLineString(
            lines,
            dt=dt,
            properties=properties
        )

    def to_geo_interface(self, **kwargs):
        return {
            **self.__geo_interface__,
            **({'bbox': self.bounds} if kwargs.get('include_bbox') else {})
        }

    def to_pyshp(self, writer):
        formatted = [
            [list(vertex.to_float()) for vertex in shape.vertices]
            for shape in self.geoshapes
        ]
        if self.has_m and not self.has_z:
            return writer.linem(formatted)
        if self.has_z:
            return writer.linez(formatted)
        return writer.line(formatted)

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


class MultiGeoPoint(MultiShapeBase, PointLikeMixin, SimpleShapeMixin):

    def __init__(
        self,
        geoshapes: List[GeoPoint],
        dt: Optional[GEOTIME_TYPE] = None,
        properties: Optional[Dict] = None,
    ):
        super().__init__(dt, properties)
        self.geoshapes: List[GeoPoint] = geoshapes

    @property
    def __geo_interface__(self):
        return {
            'type': 'MultiPoint',
            'coordinates': [
                list(point.centroid.to_float())
                for point in self.geoshapes
            ]
        }

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
        if 'coordinates' in gjson:
            geom = gjson
        else:
            geom = gjson.get('geometry', {})

        if not geom.get('type') == 'MultiPoint':
            raise ValueError(
                f'Geometry represents a {geom.get("type")}; expected MultiPoint.'
            )
        points = [
            GeoPoint(Coordinate(**dict(zip(('longitude', 'latitude', 'z'), coord))))
            for coord in geom.get('coordinates', [])
        ]
        properties = gjson.get('properties', {})
        dt = None
        if time_start_property in properties or time_end_property in properties:
            # Pop time field so it doesn't remain in properties
            dt_start = properties.pop(time_start_property, None)
            dt_end = properties.pop(time_end_property, None)

            if dt_start and not dt_end:
                dt = TimeInterval.from_str(dt_start, dt_start, time_format)

            elif dt_end and not dt_start:
                dt = TimeInterval.from_str(dt_end, dt_end, time_format)

            else:
                dt = TimeInterval.from_str(dt_start, dt_end, time_format)

        return MultiGeoPoint(
            points,
            dt=dt,
            properties=properties
        )

    @classmethod
    def from_pyshp(
        cls,
        shape,
        dt: Optional[GEOTIME_TYPE] = None,
        properties: Optional[Dict] = None
    ):
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
        points = []
        z = shape.z if hasattr(shape, 'z') else None
        m = shape.m if hasattr(shape, 'm') else None
        for point in shape.__geo_interface__.get('coordinates', []):
            points.append(
                GeoPoint(
                    Coordinate(
                        *cast(Tuple[float, float], point),
                        z=z.pop(0) if z else None,
                        m=m.pop(0) if m else None,
                    )
                )
            )

        return MultiGeoPoint(points, dt=dt, properties=properties)

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

        coords = cls._parse_wkt_linear_ring(wkt_str, _RE_LINEAR_RING.findall(wkt_str)[0])
        shapes = [
            GeoPoint(coord) for coord in coords
        ]

        return MultiGeoPoint(
            shapes,
            dt=dt,
            properties=properties
        )

    def to_geo_interface(self, **kwargs):
        return {
            **self.__geo_interface__,
            **({'bbox': self.bounds} if kwargs.get('include_bbox') else {})
        }

    def to_pyshp(self, writer):
        formatted = [x.centroid.to_float() for x in self.geoshapes]
        if self.has_m and not self.has_z:
            return writer.multipointm(formatted)
        if self.has_z:
            return writer.multipointz(formatted)
        return writer.multipoint(formatted)

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


class MultiGeoPolygon(MultiShapeBase, PolygonLikeMixin, SimpleShapeMixin):

    def __init__(
        self,
        geoshapes: Sequence[PolygonBase],
        dt: Optional[GEOTIME_TYPE] = None,
        properties: Optional[Dict] = None,
    ):
        super().__init__(dt, properties)
        self.geoshapes = list(geoshapes)

    @property
    def __geo_interface__(self):
        return {
            'type': 'MultiPolygon',
            'coordinates': [
                [
                    [
                        list(coord.to_float()) for coord in ring
                    ] for ring in shape
                ]
                for shape in self.linear_rings()
            ]
        }

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

    def copy(self) -> 'MultiGeoPolygon':
        return MultiGeoPolygon(
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
        if 'coordinates' in gjson:
            geom = gjson
        else:
            geom = gjson.get('geometry', {})

        if not geom.get('type') == 'MultiPolygon':
            raise ValueError(
                f'Geometry represents a {geom.get("type")}; expected MultiPolygon.'
            )

        shapes = []
        for shape in geom.get('coordinates', []):
            rings = [
                [Coordinate(**dict(zip(('longitude', 'latitude', 'z'), x))) for x in ring]
                for ring in shape
            ]
            shell, holes = rings[0], None
            if len(rings) > 1:
                holes = [GeoPolygon(list(reversed(ring))) for ring in rings[1:]]

            shapes.append(GeoPolygon(shell, holes=holes))

        properties = gjson.get('properties', {})
        dt = None
        if time_start_property in properties or time_end_property in properties:
            # Pop time field so it doesn't remain in properties
            dt_start = properties.pop(time_start_property, None)
            dt_end = properties.pop(time_end_property, None)

            if dt_start and not dt_end:
                dt = TimeInterval.from_str(dt_start, dt_start, time_format)

            elif dt_end and not dt_start:
                dt = TimeInterval.from_str(dt_end, dt_end, time_format)

            else:
                dt = TimeInterval.from_str(dt_start, dt_end, time_format)

        return MultiGeoPolygon(
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
        shapes = []
        z = shape.z if hasattr(shape, 'z') else None
        m = shape.m if hasattr(shape, 'm') else None
        for shape in shape.__geo_interface__.get('coordinates', []):
            linear_rings = [
                [
                    Coordinate(
                        *cast(Tuple[float, float], coord),
                        z=z.pop(0) if z else None,
                        m=m.pop(0) if m else None,
                    ) for coord in linear_ring
                ] for linear_ring in shape
            ]
            holes = [GeoPolygon(x) for x in linear_rings[1:]] if len(linear_rings) > 1 else None
            shapes.append(
                GeoPolygon(
                    linear_rings[0],
                    holes=holes
                )
            )

        return MultiGeoPolygon(shapes, dt=dt, properties=properties)

    @classmethod
    def from_wkt(
        cls,
        wkt_str: str,
        dt: Optional[GEOTIME_TYPE] = None,
        properties: Optional[Dict] = None
    ) -> 'MultiGeoPolygon':
        """Create a GeoPolygon from a wkt string"""
        if not _RE_MULTIPOLYGON_WKT.match(wkt_str):
            raise ValueError(f'Invalid WKT MultiPolygon: {wkt_str}')

        shapes = []
        for shape in _RE_LINEAR_RINGS.findall(wkt_str):
            linear_rings = _RE_LINEAR_RING.findall(shape)
            coords = cls._parse_wkt_linear_ring(wkt_str, linear_rings[0])

            holes = []
            if len(linear_rings) > 1:
                for hole in linear_rings[1:]:
                    coords = cls._parse_wkt_linear_ring(wkt_str, hole)
                    holes.append(GeoPolygon(coords))

            shapes.append(GeoPolygon(coords, holes=holes or None))

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
            poly.linear_rings(**kwargs) for poly in self.geoshapes
        ]

    def to_geo_interface(self, **kwargs):
        return {
            'type': 'MultiPolygon',
            'coordinates': [
                [
                    [
                        list(coord.to_float()) for coord in ring
                    ] for ring in shape
                ]
                for shape in self.linear_rings(k=kwargs.pop('k', None))
            ],
            **({'bbox': self.bounds} if kwargs.get('include_bbox') else {})
        }

    def to_pyshp(self, writer):
        # Note: ESRI defines right hand rule as opposite of GeoJSON
        formatted = [
            [list(coord.to_float()) for coord in ring[::-1]]
            for shape in self.geoshapes
            for ring in shape.linear_rings()
        ]
        if self.has_m and not self.has_z:
            return writer.polym(formatted)
        if self.has_z:
            return writer.polyz(formatted)
        return writer.poly(formatted)

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

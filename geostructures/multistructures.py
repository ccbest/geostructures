

__all__ = ['MultiGeoPolygon', 'MultiGeoLineString', 'MultiGeoPoint']


from abc import ABC
import copy
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from geostructures._base import (
    _GEOTIME_TYPE, _RE_COORD, _RE_MULTIPOLYGON_WKT, _RE_MULTIPOINT_WKT,
    _RE_MULTILINESTRING_WKT, _RE_LINEAR_RING, _RE_LINEAR_RINGS, _SHAPE_TYPE,
    BaseShape, MultiShapeMixin, parse_wkt_linear_ring
)
from geostructures.calc import do_edges_intersect, haversine_distance_meters
from geostructures.coordinates import Coordinate
from geostructures.structures import GeoCircle, GeoLineString, GeoPoint, GeoPolygon
from geostructures.utils.functions import is_sub_list, get_dt_from_geojson_props


class MultiGeoLineString(BaseShape, MultiShapeMixin):

    def __init__(
        self,
        linestrings: List[GeoLineString],
        dt: Optional[_GEOTIME_TYPE] = None,
        properties: Optional[Dict] = None,
    ):
        super().__init__(dt, properties)
        self.geoshapes: List[GeoLineString] = linestrings

    def __contains__(self, other: Union[BaseShape, Coordinate]):
        """Test whether a coordinate or GeoShape is contained within this geoshape"""
        if isinstance(other, Coordinate):
            return self.contains_coordinate(other)

        if other.dt is None or self.dt is None:
            return self.contains_coordinate(other.centroid)

        return self.contains_time(other.dt) and self.contains_shape(other.centroid)

    def __hash__(self) -> int:
        return hash((tuple(hash(x) for x in self.geoshapes), self.dt))

    def __repr__(self):
        pl = "s" if len(self.geoshapes) != 1 else ""
        return f'<MultiGeoLineString of {len(self.geoshapes)} linestring{pl}>'

    def area(self) -> float:
        return 0.

    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        min_lons, max_lons, min_lats, max_lats = list(
            zip(*[[x for pair in shape.bounds for x in pair] for shape in self.geoshapes])
        )
        return (min(min_lons), max(max_lons)), (min(min_lats), max(max_lats))

    @cached_property
    def centroid(self):
        # TODO: weighted by line length
        lon, lat = np.mean(
            np.array([coord.to_float() for shape in self.geoshapes for coord in shape.vertices]),
            axis=0
        )
        return Coordinate(lon, lat)

    def circumscribing_circle(self) -> 'GeoCircle':
        centroid = self.centroid
        max_dist = max(
            haversine_distance_meters(coord, centroid)
            for shape in self.geoshapes
            for coord in shape.vertices
        )
        return GeoCircle(centroid, max_dist, dt=self.dt)

    def contains_coordinate(self, coord: Coordinate) -> bool:
        for shape in self.geoshapes:
            if coord in shape.vertices:
                return True

        return False

    def contains_shape(self, shape: 'BaseShape', **kwargs) -> bool:
        if isinstance(shape, MultiShapeMixin):
            if all(self.contains_shape(subshape, **kwargs) for subshape in shape.geoshapes):
                return True
            return False

        for self_shape in self.geoshapes:
            if not self_shape.contains_shape(shape):
                return False

            return True

        return False

    def copy(self: _SHAPE_TYPE) -> _SHAPE_TYPE:
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
    def from_shapely(
        cls,
        multilinestring
    ):
        """
        Creates a GeoPolygon from a shapely polygon

        Args:
            multipoint:
                A shapely multipoint

        Returns:
            GeoPolygon
        """
        return cls.from_wkt(multilinestring.wkt)

    @classmethod
    def from_wkt(
        cls,
        wkt_str: str,
        dt: Optional[_GEOTIME_TYPE] = None,
        properties: Optional[Dict] = None
    ) -> 'MultiGeoLineString':
        """Create a GeoPolygon from a wkt string"""
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

    def intersects_shape(self, shape: 'BaseShape', **kwargs) -> bool:
        if isinstance(shape, MultiShapeMixin):
            for subshape in shape.geoshapes:
                if self.intersects_shape(subshape, **kwargs):
                    return True
            return False

        for self_shape in self.geoshapes:
            if self_shape.intersects_shape(shape):
                return True

            return False

        return False

    def linear_rings(self, **kwargs) -> List[List[List[Coordinate]]]:
        raise NotImplementedError("Points are not comprised of linear rings.")

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
                'type': 'MultiLineString',
                'coordinates': [
                    [
                        coord.to_float() for coord in line.vertices
                    ]
                    for line in self.geoshapes
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
        lines = [self._linear_ring_to_wkt(shape.vertices) for shape in self.geoshapes]
        return f'MULTILINESTRING({", ".join(lines)})'


class MultiGeoPoint(BaseShape, MultiShapeMixin):

    def __init__(
        self,
        points: List[GeoPoint],
        dt: Optional[_GEOTIME_TYPE] = None,
        properties: Optional[Dict] = None,
    ):
        super().__init__(dt, properties)
        self.geoshapes = points

    def __contains__(self, other: Union[BaseShape, Coordinate]):
        """Test whether a coordinate or GeoShape is contained within this geoshape"""
        if isinstance(other, Coordinate):
            return self.contains_coordinate(other)

        if other.dt is None or self.dt is None:
            return self.contains_coordinate(other.centroid)

        return self.contains_shape(other.centroid) and self.contains_time(other.dt)

    def __hash__(self) -> int:
        return hash(tuple(hash(x) for x in self.geoshapes))

    def __repr__(self):
        pl = "s" if len(self.geoshapes) != 1 else ""
        return f'<MultiGeoPolygon of {len(self.geoshapes)} polygon{pl}>'

    def area(self) -> float:
        return 0.

    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        min_lons, max_lons, min_lats, max_lats = list(
            zip(*[[x for pair in shape.bounds for x in pair] for shape in self.geoshapes])
        )
        return (min(min_lons), max(max_lons)), (min(min_lats), max(max_lats))

    @cached_property
    def centroid(self):
        return Coordinate(*np.average(
            np.array([point.centroid.to_float() for point in self.geoshapes])
        ))

    def bounding_coords(self, **kwargs) -> List[List[Coordinate]]:
        raise NotImplementedError('Points are not bounded')

    def bounding_edges(self, **kwargs) -> List[List[Tuple[Coordinate, Coordinate]]]:
        raise NotImplementedError('Points are not bounded')

    def circumscribing_circle(self) -> 'GeoCircle':
        centroid = self.centroid
        max_dist = max(
            haversine_distance_meters(point.centroid, centroid)
            for point in self.geoshapes
        )
        return GeoCircle(centroid, max_dist, dt=self.dt)

    def contains_coordinate(self, coord: Coordinate) -> bool:
        for point in self.geoshapes:
            if point.centroid == coord:
                return True

        return False

    def contains_shape(self, shape: 'BaseShape', **kwargs) -> bool:
        return False

    def copy(self: _SHAPE_TYPE) -> _SHAPE_TYPE:
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
                f'Geometry represents a {geom.get("type")}; expected MultiPolygon.'
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
        dt: Optional[_GEOTIME_TYPE] = None,
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

    def intersects_shape(self, shape: 'BaseShape', **kwargs) -> bool:
        if any(x.intersects_shape(shape) for x in self.geoshapes):
            return True

        return False

    def linear_rings(self, **kwargs) -> List[List[List[Coordinate]]]:
        raise NotImplementedError("Points are not comprised of linear rings.")

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
                'type': 'MultiPoint',
                'coordinates': [
                    list(point.centroid.to_float())
                    for point in self.geoshapes
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


class MultiGeoPolygon(BaseShape, MultiShapeMixin):

    def __init__(
        self,
        polygons: List[GeoPolygon],
        dt: Optional[_GEOTIME_TYPE] = None,
        properties: Optional[Dict] = None,
    ):
        super().__init__(dt, properties)
        self.geoshapes = polygons

    def __contains__(self, other: Union[BaseShape, Coordinate]):
        """Test whether a coordinate or GeoShape is contained within this geoshape"""
        if isinstance(other, Coordinate):
            return self.contains_coordinate(other)

        if other.dt is None or self.dt is None:
            return self.contains_coordinate(other.centroid)

        return self.contains_shape(other.centroid) and self.contains_time(other.dt)

    def __hash__(self) -> int:
        return hash(tuple(hash(x) for x in self.geoshapes))

    def __repr__(self):
        pl = "s" if len(self.geoshapes) != 1 else ""
        return f'<MultiGeoPolygon of {len(self.geoshapes)} polygon{pl}>'

    def area(self) -> float:
        return sum(x.area for x in self.geoshapes)

    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        min_lons, max_lons, min_lats, max_lats = list(
            zip(*[[x for pair in shape.bounds for x in pair] for shape in self.geoshapes])
        )
        return (min(min_lons), max(max_lons)), (min(min_lats), max(max_lats))

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
        return [
            [
                list(zip(ring, [*ring[1:], ring[0]]))
                for ring in shape
            ]
            for shape in self.linear_rings(**kwargs)
        ]

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

    def intersects_shape(self, shape: 'BaseShape', **kwargs) -> bool:
        if any(x.intersects_shape(shape) for x in self.geoshapes):
            return True

        return False

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

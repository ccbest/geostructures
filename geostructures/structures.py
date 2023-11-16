# pylint: disable=C0302
"""
Geospatial shape representations, for use with earth-surface calculations
"""

__all__ = [
    'GeoBox', 'GeoCircle', 'GeoEllipse', 'GeoLineString', 'GeoPoint', 'GeoPolygon',
    'GeoRing', 'GeoShape'
]

from abc import abstractmethod
import copy
from datetime import datetime
import math
import re
import statistics
from typing import cast, Any, Dict, List, Optional, Tuple, Union

import numpy as np

from geostructures.coordinates import Coordinate
from geostructures.calc import (
    ensure_vertex_bounds,
    inverse_haversine_radians,
    inverse_haversine_degrees,
    haversine_distance_meters,
    bearing_degrees,
    find_line_intersection,
    do_vertices_intersect
)
from geostructures.utils.functions import round_half_up
from geostructures.utils.mixins import LoggingMixin, DefaultZuluMixin
from geostructures.time import TimeInterval


_GEOTIME_TYPE = Union[datetime, TimeInterval]

_RE_COORD_STR = r'((?:\s?\-?\d+\.?\d*\s\-?\d+\.?\d*\s?\,?)+)'
_RE_COORD = re.compile(_RE_COORD_STR)
_RE_COORD_GROUPS_STR = r'(?:\(' + _RE_COORD_STR + r'\)\,?\s?)+'
_RE_POINT_WKT = re.compile(r'POINT\s?\((\s?\d+\.?\d*\s\d+\.?\d*\s?)\)')
_RE_POLYGON_WKT = re.compile(r'POLYGON\s?\(' + _RE_COORD_GROUPS_STR + r'\)')
_RE_LINESTRING_WKT = re.compile(r'LINESTRING\s?' + _RE_COORD_GROUPS_STR + r'\s?')


def _get_dt_from_geojson_props(
    rec: Dict[str, Any],
    time_start_field: str = 'datetime_start',
    time_end_field: str = 'datetime_end',
    time_format: Optional[str] = None
) -> Union[datetime, TimeInterval, None]:
    """Grabs datetime data and returns appropriate struct"""
    def _convert(dt: Optional[str], _format: Optional[str] = None):
        if not dt:
            return

        if _format:
            return datetime.strptime(dt, _format)

        return datetime.fromisoformat(dt)

    # Pop the field so it doesn't remain in properties
    dt_start = _convert(rec.pop(time_start_field, None), time_format)
    dt_end = _convert(rec.pop(time_end_field, None), time_format)

    if dt_start is None and dt_end is None:
        return None

    if not (dt_start and dt_end) or dt_start == dt_end:
        return dt_start or dt_end

    return TimeInterval(dt_start, dt_end)


def _parse_wkt_coord_group(group: str) -> List[Coordinate]:
    """Parse wkt coordinate list into Coordinate objects"""
    return [
        Coordinate(*coord.strip().split(' '))  # type: ignore
        for coord in group.split(',') if coord
    ]


class GeoShape(LoggingMixin, DefaultZuluMixin):

    """Abstract base class for all geoshapes"""

    def __init__(
            self,
            holes: Optional[List['GeoShape']] = None,
            dt: Optional[_GEOTIME_TYPE] = None,
            properties: Optional[Dict] = None
    ):
        super().__init__()
        if isinstance(dt, datetime):
            # Convert to a zero-second time interval
            dt = self._default_to_zulu(dt)
            self.dt: Optional[TimeInterval] = TimeInterval(dt, dt)
        else:
            self.dt = dt

        self._properties = properties or {}
        self._shapely = None
        self.holes = holes or []
        if any(x.holes for x in self.holes):
            raise ValueError('Holes cannot themselves contain holes.')

    def __contains__(self, point: Union[Coordinate, 'GeoPoint']):
        """Test whether a coordinate or GeoPoint is contained within this geoshape"""
        if isinstance(point, Coordinate):
            return self.contains_coordinate(point)

        if point.dt is None or self.dt is None:
            return self.contains_coordinate(point.centroid)

        return self.contains_coordinate(point.centroid) and self.contains_time(point.dt)

    @abstractmethod
    def __hash__(self) -> int:
        """Create unique hash of this object"""

    @abstractmethod
    def __repr__(self):
        """REPL representation of this object"""

    @property
    def area(self):
        """
        The area of the shape, in meters squared.

        Returns:
            float
        """
        from pyproj import Geod
        geod = Geod(ellps="WGS84")
        area, _ = geod.geometry_area_perimeter(self.to_shapely())
        return area

    @property
    @abstractmethod
    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        The longitude and latitude min/max bounds of the shape.

        Returns:
            ( (min_longitude, max_longitude), (min_latitude, max_latitude) )
        """

    @property
    @abstractmethod
    def centroid(self):
        """
        The center of the shape.

        Returns:
            Coordinate
        """

    @property
    def end(self) -> datetime:
        """The end date/datetime, if present"""
        if not self.dt:
            raise ValueError("GeoShape has no associated time information.")

        return self.dt.end

    @property
    def properties(self):
        props = self._properties.copy()
        if self.dt:
            props['datetime_start'] = self.start
            props['datetime_end'] = self.end

        return props

    @property
    def start(self) -> datetime:
        """The start date/datetime, if present"""
        if not self.dt:
            raise ValueError("GeoShape has no associated time information.")

        return self.dt.start

    @property
    def volume(self) -> float:
        """
        The volume of the shape, in meters squared seconds.

        Shapes with no time dimension (dt is None), or whose
        time dimension is an instance in time (dt is a datetime)
        will always have a volume of zero.

        Returns:
            float
        """
        if self.dt is None:
            return 0.

        return self.area * self.dt.elapsed.total_seconds()

    def _dt_to_json(self) -> Dict[str, str]:
        """Safely convert time bounds to json"""
        if not self.dt:
            return {}

        return {
            'datetime_start': self.start.isoformat(),
            'datetime_end': self.end.isoformat(),
        }

    @staticmethod
    def _linear_ring_to_wkt(ring: List[Coordinate]):
        """
        Converts a list of coordinates (a linear ring, self-closing) into
        a wkt string.

        Args:
            ring:
                A list of Coordinates

        Returns:
            The wkt-formatted string,
        """
        return f'({",".join(" ".join(coord.to_str()) for coord in ring)})'

    @abstractmethod
    def bounding_coords(self, **kwargs) -> List[Coordinate]:
        """
        Produces a list of coordinates that define the polygon's boundary.

        Using discrete coordinates to represent a continuous curve implies some level of data
        loss. You can minimize this loss by specifying k, which represents the number of
        points drawn.

        Does not include information about holes - see the linear rings method.

        Keyword Args:
            k: (int)
                For shapes with smooth curves, increasing k increases the number of
                points generated along the curve

        Returns:
            A list of coordinates, representing the boundary.
        """

    def bounding_vertices(self, **kwargs) -> List[Tuple[Coordinate, Coordinate]]:
        """
        Returns a list of vertices, defined as a 2-tuple (start and end) of coordinates, that
        represent the polygon's boundary.

        Using discrete coordinates to represent a continuous curve implies some level of data
        loss. You can minimize this loss by specifying k, which represents the number of
        points drawn.

        Does not include information about holes - see the vertices method.

        Keyword Args:
            k: (int)
                For shapes with smooth curves, increasing k increases the number of
                points generated along the curve

        Returns:
            A list of 2-tuples representing the vertices
        """
        bounding_coords = self.bounding_coords(**kwargs)
        return list(zip(bounding_coords, [*bounding_coords[1:], bounding_coords[0]]))

    @abstractmethod
    def circumscribing_circle(self):
        """
        Produces a circle that entirely encompasses the shape

        Returns:
            (GeoCircle)
        """

    def circumscribing_rectangle(self):
        """
        Produces a rectangle that entirely encompasses the shape

        Returns:
            (GeoBox)
        """
        lon_bounds, lat_bounds = self.bounds
        return GeoBox(
            Coordinate(lon_bounds[0], lat_bounds[1]),
            Coordinate(lon_bounds[1], lat_bounds[0]),
            dt=self.dt,
        )

    def contains(self, shape: 'GeoShape', **kwargs) -> bool:
        """
        Tests whether this shape fully contains another one.

        Args:
            shape:
                A geoshape

        Keyword Args:
            k: (int)
                For shapes with smooth curves, increasing k increases the number of
                points generated along the curve

        Returns:
            bool
        """
        # Make sure the times overlap, if present on both
        if self.dt and shape.dt:
            if not self.contains_time(shape.dt):
                return False

        s_vertices = self.vertices(**kwargs)
        o_vertices = shape.vertices(**kwargs)
        if do_vertices_intersect(
            [x for vertex_ring in s_vertices for x in vertex_ring],
            [x for vertex_ring in o_vertices for x in vertex_ring]
        ):
            # At least one vertex pair intersects - cannot be contained
            return False

        # No vertices intersect, so make sure one point along the boundary is
        # contained
        return o_vertices[0][0][0] in self

    @abstractmethod
    def contains_coordinate(self, coord: Coordinate) -> bool:
        """
        Test if a geoshape contains a coordinate.

        Args:
            coord:
                A Coordinate

        Returns:
            bool
        """

    def contains_time(self, dt: Union[datetime, TimeInterval]) -> bool:
        """
        Test if the geoshape's time dimension fully contains either a date or a datetime.

        Args:
            dt:
                A date or a datetime.

        Returns:
            bool
        """
        if self.dt is None:
            return False

        return dt in self.dt

    @abstractmethod
    def copy(self):
        """Produces a copy of the geoshape."""

    def intersects(self, shape: 'GeoShape', **kwargs) -> bool:
        """
        Tests whether another shape intersects this one.

        Args:
            shape:
                A geoshape

        Keyword Args:
            k: (int)
                For shapes with smooth curves, increasing k increases the number of
                points generated along the curve

        Returns:

        """
        # Make sure the times overlap, if present on both
        if self.dt and shape.dt:
            if not self.intersects_time(shape.dt):
                return False

        if isinstance(shape, GeoPoint):
            return shape in self

        s_vertices = self.vertices(**kwargs)
        o_vertices = shape.vertices(**kwargs)
        if do_vertices_intersect(
            [x for vertex_ring in s_vertices for x in vertex_ring],
            [x for vertex_ring in o_vertices for x in vertex_ring]
        ):
            # At least one vertex pair intersects
            return True

        # If no vertices intersect, one shape could still contain the other
        # which counts as intersection. Have to use a point from the boundary
        # because the centroid may fall in a hole
        return o_vertices[0][0][0] in self or s_vertices[0][0][0] in shape

    def intersects_time(self, dt: Union[datetime, TimeInterval]) -> bool:
        """
        Test if the geoshape's time dimension intersects either a date or a datetime.

        Args:
            dt:
                A date or a datetime.

        Returns:
            bool
        """
        if self.dt is None:
            return False

        return self.dt.intersects(dt)

    def linear_rings(self, **kwargs) -> List[List[Coordinate]]:
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
            self.bounding_coords(**kwargs),
            *[list(reversed(shape.bounding_coords())) for shape in self.holes]
        ]

    def set_property(self, key: str, value: Any):
        """
        Sets the value of a property on this geoshape.

        Args:
            key:
                The property name

            value:
                The property value

        Returns:
            None
        """
        self._properties[key] = value

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
                'type': 'Polygon',
                'coordinates': [
                    [list(coord.to_float()) for coord in ring]
                    for ring in self.linear_rings(k=k)
                ]
            },
            'properties': {
                **self.properties,
                **self._dt_to_json(),
                **(properties or {})
            },
            **kwargs
        }

    @abstractmethod
    def to_polygon(self, **kwargs):
        """
        Converts the shape to a GeoPolygon

        Returns:
            (GeoPolygon)
        """

    def to_shapely(self):
        """
        Converts the geoshape into a Shapely shape.
        """
        import shapely  # pylint: disable=import-outside-toplevel
        rings = self.linear_rings()
        holes = []
        if len(rings) > 1:
            holes = rings[1:]

        return shapely.geometry.Polygon(
            [x.to_float() for x in rings[0]],
            holes=[[x.to_float() for x in ring] for ring in holes]
        )

    def to_wkt(self, **kwargs):
        """
        Converts the shape to its WKT string representation

        Keyword Args:
            Arguments to be passed to the .bounding_coords() method. Reference
            that method for a list of corresponding kwargs.

        Returns:
            str
        """
        bbox_str = ', '.join(
            [self._linear_ring_to_wkt(ring) for ring in self.linear_rings(**kwargs)]
        )
        return f'POLYGON({bbox_str})'

    def vertices(self, **kwargs) -> List[List[Tuple[Coordinate, Coordinate]]]:
        """
        Returns lists of vertices, defined as a 2-tuple (start and end) of coordinates, for the
        outer boundary as well as holes in the polygon.

        Operates similar to the `bounding_vertices` method but includes information about holes
        in the shape.

        Using discrete coordinates to represent a continuous curve implies some level of data
        loss. You can minimize this loss by specifying k, which represents the number of
        points drawn.

        Does not include information about holes - see the vertices method.

        Keyword Args:
            k: (int)
                For shapes with smooth curves, increasing k increases the number of
                points generated along the curve

        Returns:
            Lists of 2-tuples representing vertices. The first list will always represent
            the shape's boundary, and any following lists will represent holes.
        """
        rings = self.linear_rings(**kwargs)
        return [
            list(zip(ring, ring[1:]))
            for ring in rings
        ]


class GeoPolygon(GeoShape):

    """
    A Polygon, as expressed by an ordered list of Coordinates. The final Coordinate
    must be identical to the first Coordinate.

    Args:
        outline: (List[Coordinate])
            A list of coordinates that define the outside edges of the polygon

        holes: (List[Coordinate])
            Additional lists of coordinates representing holes in the polygon

        dt: (datetime | TimeInterval | None)


        properties: dict
            Additional properties that describe this geoshape.

    """

    def __init__(
        self,
        outline: List[Coordinate],
        holes: Optional[List[GeoShape]] = None,
        dt: Optional[_GEOTIME_TYPE] = None,
        properties: Optional[Dict] = None,
        _is_hole: bool = False,
    ):
        super().__init__(holes=holes, dt=dt, properties=properties)

        if not outline[0] == outline[-1]:
            self.logger.warning(
                'Polygon outlines must be self-closing; your final point will be '
                'connected to your starting point.'
            )
            outline = [*outline, outline[0]]

        if not self._test_counter_clockwise(outline) ^ _is_hole:
            self.warn_once(
                'Your polygon appears to be defined (mostly) clockwise, violating the '
                'right hand rule. Flipping coordinate order; this warning will not repeat.'
            )
            outline = outline[::-1]

        self.outline = outline

    def __eq__(self, other):
        if not isinstance(other, GeoPolygon):
            return False

        if not self.dt == other.dt:
            return False

        # Determine if self.outline matches any (forward or backward) rotation of
        # other.outline
        if len(self.outline) != len(other.outline):
            return False  # Can't match if not the same number of points

        s_outline = self.outline[0:-1]
        o_outline = other.outline[0:-1]
        outline_eq = False
        for _ in range(0, len(o_outline)):
            # Rotate the outline
            if s_outline in (o_outline, o_outline[::-1]):
                outline_eq = True
                break
            o_outline = o_outline[1:] + [o_outline[0]]

        if not outline_eq:
            return False

        if len(self.holes) != len(other.holes):
            return False

        s_holes = set(
            [
                tuple({(x, y) for x, y in zip(hole.bounding_coords(), hole.bounding_coords()[1:])})
                for hole in self.holes
            ]
        )
        o_holes = set(
            [
                tuple({(x, y) for x, y in zip(hole.bounding_coords(), hole.bounding_coords()[1:])})
                for hole in other.holes
            ]
        )
        return s_holes == o_holes

    def __hash__(self):
        return hash((tuple(self.outline), self.dt))

    def __repr__(self):
        return f'<GeoPolygon of {len(self.outline) - 1} coordinates>'

    @property
    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        lons, lats = cast(
            Tuple[List[float], List[float]],
            zip(*[y.to_float() for y in self.outline])
        )
        return (min(lons), max(lons)), (min(lats), max(lats))

    @property
    def centroid(self):
        # Decompose polygon into triangles using vertex pairs around the origin
        poly1 = np.array([x.to_float() for x in self.bounding_coords()])
        poly2 = np.roll(poly1, -1, axis=0)
        # Find signed area of each triangle
        signed_areas = 0.5 * np.cross(poly1, poly2)
        # Return average of triangle centroids, weighted by area
        return Coordinate(*np.average((poly1 + poly2) / 3, axis=0, weights=signed_areas))

    @staticmethod
    def _point_in_polygon(
            coord: Coordinate,
            polygon: List[Coordinate],
            include_boundary: bool = False,
    ) -> bool:
        """
        Tests whether a point is in the polygon. From the point, draws
        a straight line along the latitude and determines how many sides
        of the polygon intersect with it. If there are an odd number of
        intersections, the point is in the polygon.

        Notes:
            Use __contains__ instead of calling this function directly,
            because it handles common-sense methods to filter out points
            that are definitely not in the polygon.

        Args:
            coord:
                A Coordinate

            polygon:
                A list of coordinates (self-closing) representing a linear ring

            include_boundary:
                Whether to count boundaries as intersecting (parallel overlapping
                lines still do not count)

        Returns:
            bool
        """
        test_line = (coord, Coordinate(180, float(coord.latitude)))
        _intersections = 0
        for vertex in zip(polygon, [*polygon[1:], polygon[0]]):
            intersection = find_line_intersection(test_line, vertex)
            if not intersection:
                continue

            if intersection[1] and not include_boundary:
                # Lies on boundary, no need to continue
                return False

            if include_boundary or not intersection[1]:
                # If boundaries are allowed, or is not a boundary intersection
                _intersections += 1

        return _intersections > 0 and _intersections % 2 != 0

    @staticmethod
    def _test_counter_clockwise(bounds: List[Coordinate]) -> bool:
        """
        Tests a polygon to determine whether it's defined in a counterclockwise
        (or mostly, for complex shapes) order.

        Args:
            bounds:
                A list of Coordinates, in order

        Returns:
            bool
        """
        ans = sum(
            (y.longitude - x.longitude) * (y.latitude + x.latitude)
            for x, y in map(
                lambda x: ensure_vertex_bounds(x[0], x[1]),
                zip(bounds, [*bounds[1:], bounds[0]])
            )
        )
        return ans <= 0

    def bounding_coords(self, **kwargs) -> List[Coordinate]:
        return self.outline

    def circumscribing_circle(self):
        centroid = self.centroid
        max_dist = max(
            haversine_distance_meters(x, centroid) for x in self.outline[:-1]
        )
        return GeoCircle(centroid, max_dist, dt=self.dt)

    def contains_coordinate(self, coord: Coordinate) -> bool:
        # First see if the point even falls inside the circumscribing rectangle
        lon_bounds, lat_bounds = self.bounds
        if not (
            lon_bounds[0] <= coord.longitude <= lon_bounds[1] and
            lat_bounds[0] <= coord.latitude <= lat_bounds[1]
        ):
            # Falls outside rectangle - not in polygon
            return False

        # If not inside outline, no need to continue
        if not self._point_in_polygon(coord, self.outline):
            return False

        for hole in self.holes:
            if coord in hole:
                return False

        return True

    def copy(self):
        return GeoPolygon(
            self.outline.copy(),
            holes=self.holes.copy(),
            dt=self.dt.copy() if self.dt else None,
            properties=copy.deepcopy(self.properties)
        )

    @classmethod
    def from_geojson(
        cls,
        gjson: Dict[str, Any],
        time_start_property: str = 'datetime_start',
        time_end_property: str = 'datetime_end',
        time_format: Optional[str] = None,
    ):
        """
        Creates a GeoPolygon from a GeoJSON polygon.

        Args:
            gjson:
                A geojson dictionary

            time_start_property:
                The geojson property containing the start time, if available

            time_end_property:
                The geojson property containing hte ned time, if available

            time_format: (Optional)
                The format of the timestamps in the above time fields.

        Returns:
            GeoPolygon
        """
        geom = gjson.get('geometry', {})
        if not geom.get('type') == 'Polygon':
            raise ValueError(
                f'Geometry represents a {geom.get("type")}; expected Polygon.'
            )

        rings = [[Coordinate(x, y) for x, y in ring] for ring in geom.get('coordinates', [])]
        holes: List[GeoShape] = []
        if len(rings) > 1:
            holes = [GeoPolygon(ring) for ring in rings[1:]]

        properties = gjson.get('properties', {})
        dt = _get_dt_from_geojson_props(
            properties,
            time_start_property,
            time_end_property,
            time_format
        )

        return GeoPolygon(rings[0], holes=holes, dt=dt, properties=properties)

    @classmethod
    def from_shapely(
        cls,
        polygon
    ):
        """
        Creates a GeoPolygon from a shapely polygon

        Args:
            polygon:
                A shapely polygon

        Returns:
            GeoPolygon
        """

        return cls.from_wkt(polygon.wkt)

    @classmethod
    def from_wkt(
        cls,
        wkt_str: str,
        dt: Optional[_GEOTIME_TYPE] = None,
        properties: Optional[Dict] = None
    ):
        """Create a GeoPolygon from a wkt string"""
        if not _RE_POLYGON_WKT.match(wkt_str):
            raise ValueError(f'Invalid WKT Polygon: {wkt_str}')

        coord_groups = _RE_COORD.findall(wkt_str)
        outline = _parse_wkt_coord_group(coord_groups[0])
        holes: List[GeoShape] = []
        if len(coord_groups) > 1:
            holes = [
                GeoPolygon(_parse_wkt_coord_group(coord_group))
                for coord_group in coord_groups[1:]
            ]

        return GeoPolygon(outline, holes=holes, dt=dt, properties=properties)

    def to_wkt(self, **kwargs):
        """
        Converts the shape to its WKT string representation

        Keyword Args:
            Arguments to be passed to the .linear_rings() method. Reference
            that method for a list of corresponding kwargs.

        Returns:
            str
        """
        bboxs = [self._linear_ring_to_wkt(ring) for ring in self.linear_rings(**kwargs)]
        return f'POLYGON({",".join(bboxs)})'

    def to_polygon(self, **kwargs):
        return self


class GeoBox(GeoShape):

    """
    A Box (or Square), as expressed by the Northwest and Southeast corners.

    Args:
        nw_bound: (Coordinate)
            The Northwest corner of the box

        se_bound: (Coordinate)
            The Southeast corner of the box

    """

    def __init__(
        self,
        nw_bound: Coordinate,
        se_bound: Coordinate,
        holes: Optional[List[GeoShape]] = None,
        dt: Optional[_GEOTIME_TYPE] = None,
        properties: Optional[Dict] = None,
    ):
        super().__init__(holes=holes, dt=dt, properties=properties)
        self.nw_bound = nw_bound
        self.se_bound = se_bound

    def __eq__(self, other):
        if not isinstance(other, GeoBox):
            return False

        return (
            self.nw_bound == other.nw_bound
            and self.se_bound == other.se_bound
            and self.dt == other.dt
        )

    def __hash__(self):
        return hash((self.nw_bound, self.se_bound, self.dt))

    def __repr__(self):
        return f'<GeoBox {self.nw_bound.to_float()} - {self.se_bound.to_float()}>'

    @property
    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return (
            (self.nw_bound.longitude, self.se_bound.longitude),
            (self.se_bound.latitude, self.nw_bound.latitude)
        )

    @property
    def centroid(self):
        _nw = self.nw_bound.to_float()
        _se = self.se_bound.to_float()
        return Coordinate(
            round_half_up(statistics.mean([_nw[0], _se[0]]), 7),
            round_half_up(statistics.mean([_nw[1], _se[1]]), 7),
        )

    def bounding_coords(self, **kwargs):
        _nw = self.nw_bound.to_float()
        _se = self.se_bound.to_float()

        # Is self-closing
        return [
            self.nw_bound,
            Coordinate(_nw[0], _se[1]),
            self.se_bound,
            Coordinate(_se[0], _nw[1]),
            self.nw_bound,
        ]

    def circumscribing_rectangle(self):
        return self

    def circumscribing_circle(self):
        return GeoCircle(
            self.centroid,
            haversine_distance_meters(self.nw_bound, self.centroid),
            dt=self.dt,
        )

    def contains_coordinate(self, coord: Coordinate):
        if not (
            self.nw_bound.longitude <= coord.longitude <= self.se_bound.longitude and
            self.se_bound.latitude <= coord.latitude <= self.nw_bound.latitude
        ):
            return False

        for hole in self.holes:
            if coord in hole:
                return False

        return True

    def copy(self):
        return GeoBox(
            self.nw_bound,
            self.se_bound,
            holes=self.holes.copy(),
            dt=self.dt.copy() if self.dt else None,
            properties=copy.deepcopy(self.properties)
        )

    def to_polygon(self, **kwargs):
        return GeoPolygon(*self.linear_rings(**kwargs), dt=self.dt)


class GeoCircle(GeoShape):

    """
    A circle shape, as expressed by:
        * A Coordinate center
        * A radius

    Args:
        center: (Coordinate)
            The circle centroid

        radius: (float)
            The length of the circle's radius, in meters
    """

    def __init__(
        self,
        center: Coordinate,
        radius: float,
        holes: Optional[List[GeoShape]] = None,
        dt: Optional[_GEOTIME_TYPE] = None,
        properties: Optional[Dict] = None,
    ):
        super().__init__(holes=holes, dt=dt, properties=properties)
        self.center = center
        self.radius = radius

    def __eq__(self, other):
        if not isinstance(other, GeoCircle):
            return False

        return (
            self.center == other.center
            and self.radius == other.radius
            and self.dt == other.dt
        )

    def __hash__(self):
        return hash((self.centroid, self.radius, self.dt))

    def __repr__(self):
        return f'<GeoCircle at {self.centroid.to_float()}; radius {self.radius} meters>'

    @property
    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        nw_bound = inverse_haversine_degrees(
            self.center, 315, self.radius * math.sqrt(2)
        )
        se_bound = inverse_haversine_degrees(
            self.center, 135, self.radius * math.sqrt(2)
        )
        return (nw_bound.longitude, se_bound.longitude), (se_bound.latitude, nw_bound.latitude)

    @property
    def centroid(self):
        return self.center

    def bounding_coords(self, **kwargs) -> List[Coordinate]:
        k = kwargs.get('k') or 36
        coords = []

        for i in range(k, -1, -1):
            angle = math.pi * 2 / k * i
            coord = inverse_haversine_radians(self.center, angle, self.radius)
            coords.append(coord)

        return [*coords, coords[0]]

    def circumscribing_circle(self):
        return self

    def contains_coordinate(self, coord: Coordinate) -> bool:
        if not haversine_distance_meters(coord, self.center) <= self.radius:
            return False

        for hole in self.holes:
            if coord in hole:
                return False

        return True

    def copy(self):
        return GeoCircle(
            self.center,
            self.radius,
            holes=self.holes.copy(),
            dt=self.dt.copy() if self.dt else None,
            properties=copy.deepcopy(self.properties)
        )

    def to_polygon(self, **kwargs):
        return GeoPolygon(self.bounding_coords(**kwargs), dt=self.dt)


class GeoEllipse(GeoShape):

    """
    An ellipsoid shape (or oval), represented by:
        * a Coordinate center
        * a major axis (the radius at its greatest value)
        * a minor axis (the radius at its least value)
        * rotation (the major axis's degree offset from North)

    Args:
        center: (Coordinate)
            The centroid of the ellipse

        major_axis: (float)
            The maximum radius value

        minor_axis: (float)
            The minimum radius value

        rotation: (float)
            The major axis's degree offset from North (expressed as East of North)

    """

    def __init__(  # pylint: disable=R0913
        self,
        center: Coordinate,
        major_axis: float,
        minor_axis: float,
        rotation: float,
        holes: Optional[List[GeoShape]] = None,
        dt: Optional[_GEOTIME_TYPE] = None,
        properties: Optional[Dict] = None,
    ):
        super().__init__(holes=holes, dt=dt, properties=properties)

        self.center = center
        self.major_axis = major_axis
        self.minor_axis = minor_axis
        self.rotation = rotation

    def __eq__(self, other):
        if not isinstance(other, GeoEllipse):
            return False

        return (
            self.center == other.center
            and self.major_axis == other.major_axis
            and self.minor_axis == other.minor_axis
            and self.rotation == other.rotation
            and self.dt == other.dt
        )

    def __hash__(self):
        return hash(
            (self.centroid, self.minor_axis, self.major_axis, self.rotation, self.dt)
        )

    def __repr__(self):
        return (
            f'<GeoEllipse at {self.center.to_float()}; '
            f'radius {self.major_axis}/{self.minor_axis}; '
            f'rotation {self.rotation}>'
        )

    @property
    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        lons, lats = cast(
            Tuple[List[float], List[float]],
            zip(*[y.to_float() for y in self.bounding_coords()])
        )
        return (min(lons), max(lons)), (min(lats), max(lats))

    @property
    def centroid(self):
        return self.center

    def _radius_at_angle(self, angle: float) -> float:
        """
        Returns the ellipse radius length at a given angle.

        Args:
            angle:
                The angle of direction from the ellipse center

        Returns:
            float
        """
        return (
            self.major_axis
            * self.minor_axis
            / math.sqrt(
                (self.major_axis**2) * (math.sin(angle) ** 2)
                + (self.minor_axis**2) * (math.cos(angle) ** 2)
            )
        )

    def bounding_coords(self, **kwargs):
        k = kwargs.get('k') or math.ceil(36 * self.major_axis / self.minor_axis)
        coords = []
        rotation = math.radians(self.rotation)

        for i in range(k, -1, -1):
            angle = (math.pi * 2 / k) * i
            radius = self._radius_at_angle(angle)
            coord = inverse_haversine_radians(
                self.center, angle + rotation, radius
            )
            coords.append(coord)

        return [*coords, coords[0]]

    def circumscribing_circle(self):
        return GeoCircle(self.center, self.major_axis, dt=self.dt)

    def contains_coordinate(self, coord: Coordinate) -> bool:
        bearing = bearing_degrees(self.center, coord)
        radius = self._radius_at_angle(math.radians(bearing - self.rotation))
        if not haversine_distance_meters(self.center, coord) <= radius:
            return False

        for hole in self.holes:
            if coord in hole:
                return False

        return True

    def copy(self):
        return GeoEllipse(
            self.center,
            self.major_axis,
            self.minor_axis,
            self.rotation,
            holes=self.holes.copy(),
            dt=self.dt.copy() if self.dt else None,
            properties=copy.deepcopy(self.properties)
        )

    def to_polygon(self, **kwargs):
        return GeoPolygon(self.bounding_coords(**kwargs), dt=self.dt)


class GeoRing(GeoShape):

    """
    A ring shape consisting of the area between two concentric circles, represented by:
        * A Coordinate centroid
        * The radius length of the inner circle
        * The radius length of the outer circle

    Optionally, you may also specify minimum and maximum angles, creating a wedge shape.

    Args:
        center: (Coordinate)
            The center coordinate of the shape.

        inner_radius: (float)
            The length of the inner circle's radius, in meters

        outer_radius: (float)
            The length of the outer circle's radius, in meters

        angle_min: (float) (Optional)
            The minimum angle (expressed as degrees East of North) from which to create a
            wedge shape. If not provided, an angle of 0 degrees will be inferred.

        angle_max: (float) (Optional)
            The maximum angle (expressed as degrees East of North) from which to create a
            wedge shape. If not provided, an angle of 360 degrees will be inferred.

    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        center: Coordinate,
        inner_radius: float,
        outer_radius: float,
        angle_min: float = 0.0,
        angle_max: float = 360.0,
        holes: Optional[List[GeoShape]] = None,
        dt: Optional[_GEOTIME_TYPE] = None,
        properties: Optional[Dict] = None,
    ):
        super().__init__(holes=holes, dt=dt, properties=properties)
        self.center = center
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius

    def __eq__(self, other):
        if not isinstance(other, GeoRing):
            return False

        return (
            self.center == other.center
            and self.inner_radius == other.inner_radius
            and self.outer_radius == other.outer_radius
            and self.angle_min == other.angle_min
            and self.angle_max == other.angle_max
            and self.dt == other.dt
        )

    def __hash__(self):
        return hash(
            (
                self.centroid,
                self.inner_radius,
                self.outer_radius,
                self.angle_min,
                self.angle_max,
                self.dt,
            )
        )

    def __repr__(self):
        return (
            f'<GeoRing at {self.center.to_float()}; '
            f'radii {self.inner_radius}/{self.outer_radius}'
            f'{f"; {self.angle_min}-{self.angle_max} degrees" if self.angle_min else ""}>'
        )

    @property
    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        if self.angle_max - self.angle_min >= 360:
            nw_bound = inverse_haversine_degrees(
                self.center, 315, self.outer_radius * math.sqrt(2)
            )
            se_bound = inverse_haversine_degrees(
                self.center, 135, self.outer_radius * math.sqrt(2)
            )
            return (nw_bound.longitude, se_bound.longitude), (se_bound.latitude, nw_bound.latitude)

        lons, lats = cast(
            Tuple[List[float], List[float]],
            zip(*[y.to_float() for y in self.bounding_coords()])
        )
        return (min(lons), max(lons)), (min(lats), max(lats))

    @property
    def centroid(self):
        if self.angle_min and self.angle_max:
            return self.to_polygon().centroid

        return self.center

    def _draw_bounds(self, **kwargs):
        k = kwargs.get('k') or max(math.ceil((self.angle_max - self.angle_min) / 10), 10)
        outer_coords = []
        inner_coords = []

        for i in range(k, -1, -1):
            angle = (
                math.pi
                * (self.angle_min + (self.angle_max - self.angle_min) / k * i)
                / 180
            )
            coord = inverse_haversine_radians(
                self.center, angle, self.outer_radius
            )
            outer_coords.append(coord)

            coord = inverse_haversine_radians(
                self.center, angle, self.inner_radius
            )
            inner_coords.append(coord)

        return outer_coords, inner_coords

    def bounding_coords(self, **kwargs):
        outer_bounds, inner_bounds = self._draw_bounds(**kwargs)

        if self.angle_min == 0 and self.angle_max == 360:
            return [*outer_bounds, outer_bounds[0]]

        # Is self-closing
        return [*outer_bounds, *inner_bounds[::-1], outer_bounds[0]]

    def circumscribing_circle(self):
        if self.angle_min and self.angle_max:
            # declare as variable to avoid recomputing
            _centroid = self.centroid

            return GeoCircle(
                _centroid,
                max(
                    haversine_distance_meters(x, _centroid)
                    for x in self.bounding_coords()
                ),
                dt=self.dt,
            )

        return GeoCircle(self.centroid, self.outer_radius, dt=self.dt)

    def contains_coordinate(self, coord: Coordinate) -> bool:
        # Make sure bearing within wedge, if a wedge
        if self.angle_max - self.angle_min < 360:
            bearing = bearing_degrees(self.center, coord)
            if not self.angle_min <= bearing <= self.angle_max:
                return False

        radius = haversine_distance_meters(self.center, coord)
        if not self.inner_radius <= radius <= self.outer_radius:
            return False

        for hole in self.holes:
            if coord in hole:
                return False

        return True

    def copy(self):
        return GeoRing(
            self.center,
            self.inner_radius,
            self.outer_radius,
            self.angle_min,
            self.angle_max,
            holes=self.holes.copy(),
            dt=self.dt.copy() if self.dt else None,
            properties=copy.deepcopy(self.properties)
        )

    def linear_rings(self, **kwargs) -> List[List[Coordinate]]:
        outer_bounds, inner_bounds = self._draw_bounds(**kwargs)

        if self.angle_min == 0 and self.angle_max == 360:
            # Shape is really a circle with hole
            return [
                [*outer_bounds, outer_bounds[0]],
                list(reversed([*inner_bounds, inner_bounds[0]])),
                *[list(reversed(shape.bounding_coords(**kwargs))) for shape in self.holes]
            ]

        # Is self-closing
        return [
            [*outer_bounds, *inner_bounds[::-1], outer_bounds[0]],
            *[list(reversed(shape.bounding_coords(**kwargs))) for shape in self.holes]
        ]

    def to_polygon(self, **kwargs):
        return GeoPolygon(self.bounding_coords(**kwargs), dt=self.dt)

    def to_wkt(self, **kwargs) -> str:
        if self.angle_min == 0 and self.angle_max == 360:
            # Return as a polygon with hole
            outer_circle = GeoCircle(self.center, self.outer_radius).bounding_coords(**kwargs)
            outer_bbox_str = ",".join(
                " ".join(x.to_str()) for x in outer_circle
            )
            inner_circle = GeoCircle(self.center, self.inner_radius).bounding_coords(**kwargs)
            inner_bbox_str = ",".join(
                " ".join(x.to_str()) for x in inner_circle
            )
            return f'POLYGON(({outer_bbox_str}), ({inner_bbox_str}))'

        return super().to_wkt(**kwargs)


class GeoLineString(GeoShape):

    """
    A LineString (or more colloquially, a path) consisting of a series of
    """

    def __init__(
            self,
            coords: List[Coordinate],
            dt: Optional[_GEOTIME_TYPE] = None,
            properties: Optional[Dict] = None
    ):
        super().__init__(dt=dt, properties=properties)
        self.coords = coords

    def __eq__(self, other):
        if not isinstance(other, GeoLineString):
            return False

        return self.coords == other.coords and self.dt == other.dt

    def __hash__(self):
        return hash((tuple(self.coords), self.dt))

    def __repr__(self):
        return f'<GeoLineString with {len(self.coords)} points>'

    @property
    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        lons, lats = cast(
            Tuple[List[float], List[float]],
            zip(*[y.to_float() for y in self.coords])
        )
        return (min(lons), max(lons)), (min(lats), max(lats))

    @property
    def centroid(self):
        return Coordinate(
            *[
                round_half_up(statistics.mean(x), 7)
                for x in zip(*[y.to_float() for y in self.coords])
            ]
        )

    def bounding_coords(self, **_):
        return self.coords

    def bounding_vertices(self, **_) -> List[Tuple[Coordinate, Coordinate]]:
        return list(zip(self.bounding_coords(), self.bounding_coords()[1:]))

    def circumscribing_circle(self):
        centroid = self.centroid
        max_dist = max(haversine_distance_meters(x, centroid) for x in self.coords)
        return GeoCircle(centroid, max_dist, dt=self.dt)

    def circumscribing_rectangle(self):
        lons, lats = zip(*[y.to_float() for y in self.coords])
        return GeoBox(
            Coordinate(min(lons), max(lats)),
            Coordinate(max(lons), min(lats)),
            dt=self.dt,
        )

    def contains(self, shape: 'GeoShape', **kwargs) -> bool:
        def is_sub(sub, ls):
            """Tests if a list is a subset of another, order matters"""
            ln = len(sub)
            return any(
                (all(sub[j] == ls[i + j] for j in range(ln)) for i in range(len(ls) - ln + 1))
            )

        # Make sure the times overlap, if present on both
        if self.dt and shape.dt:
            if not self.contains_time(shape.dt):
                return False

        if not isinstance(shape, (GeoPoint, GeoLineString)):
            return False

        if isinstance(shape, GeoPoint):
            return shape in self

        return is_sub(shape.coords, self.coords)

    def contains_coordinate(self, coord: Coordinate) -> bool:
        # For now, just check for exact match. Will need update if buffering
        # becomes a feature
        return coord in self.coords

    def copy(self):
        return GeoLineString(
            self.coords,
            dt=self.dt.copy() if self.dt else None,
            properties=copy.deepcopy(self.properties)
        )

    @classmethod
    def from_geojson(
        cls,
        gjson: Dict[str, Any],
        time_start_property: str = 'datetime_start',
        time_end_property: str = 'datetime_end',
        time_format: Optional[str] = None,
    ):
        """
        Creates a GeoLineString from a GeoJSON LineString.

        Args:
            gjson:
                A geojson object, representing a linestring

            time_start_property:
                The geojson property containing the start time, if available

            time_end_property:
                The geojson property containing hte ned time, if available

            time_format: (Optional)
                The format of the timestamps in the above time fields.

        Returns:
            GeoLineString
        """
        geom = gjson.get('geometry', {})
        if not geom.get('type') == 'LineString':
            raise ValueError(
                f'Geometry represents a {geom.get("type")}; expected LineString.'
            )

        coords = [Coordinate(x, y) for x, y in geom.get('coordinates', [])]
        properties = gjson.get('properties', {})
        dt = _get_dt_from_geojson_props(
            properties,
            time_start_property,
            time_end_property,
            time_format
        )
        return GeoLineString(coords, dt=dt, properties=properties)

    @classmethod
    def from_shapely(cls, linestring):
        """
        Creates a GeoLinestring from a shapely Linestring
        Args:
            linestring:
                A shapely linestring

        Returns:
            GeoLinestring
        """
        return cls.from_wkt(linestring.wkt)

    @classmethod
    def from_wkt(
        cls,
        wkt_str: str,
        dt: Optional[_GEOTIME_TYPE] = None,
        properties: Optional[Dict] = None
    ):
        """Create a GeoLineString from a wkt string"""
        if not _RE_LINESTRING_WKT.match(wkt_str):
            raise ValueError(f'Invalid WKT LineString: {wkt_str}')

        coord_groups = _RE_COORD.findall(wkt_str)
        if not len(coord_groups) == 1:
            raise ValueError(f'Invalid WKT LineString: {wkt_str}')

        return GeoLineString(
            _parse_wkt_coord_group(coord_groups[0]),
            dt=dt,
            properties=properties
        )

    def linear_rings(self, **kwargs) -> List[List[Coordinate]]:
        raise NotImplementedError("Linestrings are not comprised of linear rings.")

    def to_geojson(
            self,
            k: Optional[int] = None,
            properties: Optional[Dict] = None,
            **kwargs
    ) -> Dict:
        return {
            'type': 'Feature',
            'geometry': {
                'type': 'LineString',
                'coordinates': [list(x.to_float()) for x in self.bounding_coords(k=k)],
            },
            'properties': {
                **self.properties,
                **self._dt_to_json(),
                **(properties or {})
            },
            **kwargs
        }

    def to_polygon(self, **kwargs):
        return GeoPolygon([*self.coords, self.coords[0]], dt=self.dt)

    def to_shapely(self):
        if self._shapely:  # pragma: no cover
            # Check if memoized
            return self._shapely

        import shapely
        self._shapely = shapely.LineString([x.to_float() for x in self.coords])
        return self._shapely

    def to_wkt(self, **kwargs):
        bbox_str = self._linear_ring_to_wkt(self.bounding_coords(**kwargs))
        return f'LINESTRING{bbox_str}'

    def vertices(self, **_) -> List[List[Tuple[Coordinate, Coordinate]]]:
        return [self.bounding_vertices()]


class GeoPoint(GeoShape):

    """
    A Coordinate with an associated timestamp. This is the only shape which
    actually requires a timestamp; if your points are time-less you should just
    use the Coordinate object.

    Args:

    """

    def __init__(
        self,
        center: Coordinate,
        dt: Optional[_GEOTIME_TYPE] = None,
        properties: Optional[Dict] = None
    ):
        super().__init__(dt=dt, properties=properties)
        self.center = center

    def __eq__(self, other):
        if not isinstance(other, GeoPoint):
            return False

        return self.center == other.center and self.dt == other.dt

    def __hash__(self):
        return hash((self.center, self.dt))

    def __repr__(self):
        return f'<GeoPoint at {self.center.to_float()}>'

    @property
    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return (
            (self.center.longitude, self.center.longitude),
            (self.center.latitude, self.center.latitude)
        )

    @property
    def centroid(self):
        return self.center

    def bounding_coords(self, **kwargs):
        raise NotImplementedError('Points are not bounded')

    def bounding_vertices(self, **kwargs):
        raise NotImplementedError('Points are not bounded')

    def circumscribing_circle(self):
        raise NotImplementedError('Points cannot be circumscribed')

    def circumscribing_rectangle(self):
        raise NotImplementedError('Points cannot be circumscribed')

    def contains(self, shape: 'GeoShape', **kwargs) -> bool:
        return False

    def contains_coordinate(self, coord: Coordinate) -> bool:
        # Points don't contain anything, even themselves
        return False

    def copy(self):
        return GeoPoint(
            self.center,
            dt=self.dt.copy() if self.dt else None,
            properties=copy.deepcopy(self.properties)
        )

    def intersects(self, shape: 'GeoShape', **kwargs) -> bool:
        if isinstance(shape, GeoPoint):
            return self == shape
        return self in shape

    @classmethod
    def from_geojson(
        cls,
        gjson: Dict[str, Any],
        time_start_property: str = 'datetime_start',
        time_end_property: str = 'datetime_end',
        time_format: Optional[str] = None,
    ):
        """
        Creates a GeoPoint from a GeoJSON point.

        Args:
            gjson:
                A geojson dictionary

            time_start_property:
                The geojson property containing the start time, if available

            time_end_property:
                The geojson property containing hte ned time, if available

            time_format: (Optional)
                The format of the timestamps in the above time fields.

        Returns:
            GeoPoint
        """
        geom = gjson.get('geometry', {})
        if not geom.get('type') == 'Point':
            raise ValueError(
                f'Geometry represents a {geom.get("type")}; expected Point.'
            )

        coordinates = geom['coordinates']
        coord = Coordinate(coordinates[0], coordinates[1])
        properties = gjson.get('properties', {})
        dt = _get_dt_from_geojson_props(
            properties,
            time_start_property,
            time_end_property,
            time_format
        )

        return GeoPoint(coord, dt=dt, properties=properties)

    @classmethod
    def from_shapely(cls, point):
        """
        Creates a GeoPoint from a shapely Point
        Args:
            point:
                A shapely Point

        Returns:
            GeoPoint
        """
        return cls.from_wkt(point.wkt)

    @classmethod
    def from_wkt(
        cls,
        wkt_str: str,
        dt: Optional[_GEOTIME_TYPE] = None,
        properties: Optional[Dict] = None
    ):
        """Create a GeoPoint from a wkt string"""
        _match = _RE_POINT_WKT.match(wkt_str)
        if not _match:
            raise ValueError(f'Invalid WKT Point: {wkt_str}')

        return GeoPoint(
            Coordinate(*_match.groups()[0].split(' ')),
            dt=dt,
            properties=properties
        )

    def linear_rings(self, **kwargs) -> List[List[Coordinate]]:
        raise NotImplementedError("Points are not comprised of linear rings.")

    def to_geojson(
            self,
            k: Optional[int] = None,
            properties: Optional[Dict] = None,
            **kwargs
    ) -> Dict:
        return {
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': list(self.center.to_float()),
            },
            'properties': {
                **self.properties,
                **self._dt_to_json(),
                **(properties or {})
            },
            **kwargs
        }

    def to_polygon(self, **kwargs):
        raise NotImplementedError('Points cannot be converted to polygons')

    def to_shapely(self):
        if self._shapely:  # pragma: no cover
            return self._shapely

        import shapely
        self._shapely = shapely.Point(self.centroid.longitude, self.centroid.latitude)
        return self._shapely

    def to_wkt(self, **_):
        return f'POINT({" ".join(self.center.to_str())})'

    def vertices(self, **kwargs):
        raise NotImplementedError('Points are not bounded')

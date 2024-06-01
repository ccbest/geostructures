# pylint: disable=C0302
"""
Geospatial shape representations, for use with earth-surface calculations
"""

__all__ = [
    'GeoBox', 'GeoCircle', 'GeoEllipse', 'GeoLineString', 'GeoPoint', 'GeoPolygon',
    'GeoRing', 'ShapeBase'
]

from abc import ABC
import copy
from functools import cached_property
import math
import statistics
from typing import cast, Any, Dict, List, Optional, Tuple, Sequence

import numpy as np

from geostructures import LOGGER
from geostructures._base import (
    _RE_COORD, _RE_LINEAR_RING, _RE_POINT_WKT, _RE_POLYGON_WKT,
    _RE_LINESTRING_WKT, BaseShape, ShapeLike, LineLike, MultiShapeBase,
    PointLike, parse_wkt_linear_ring, ANY_SHAPE_TYPE
)
from geostructures.time import GEOTIME_TYPE
from geostructures.coordinates import Coordinate
from geostructures.calc import (
    inverse_haversine_radians,
    inverse_haversine_degrees,
    haversine_distance_meters,
    bearing_degrees
)
from geostructures._geometry import circumscribing_circle_for_polygon, do_edges_intersect, find_line_intersection, \
    is_counter_clockwise
from geostructures.utils.functions import round_half_up, get_dt_from_geojson_props, is_sub_list
from geostructures.utils.logging import warn_once


class ShapeBase(BaseShape, ShapeLike, ABC):

    def __init__(
        self,
        holes: Optional[Sequence[ShapeLike]] = None,
        dt: Optional[GEOTIME_TYPE] = None,
        properties: Optional[Dict] = None,
    ):
        super().__init__(dt=dt, properties=properties)
        self.holes = list(holes or [])
        if any(x.holes for x in self.holes):
            raise ValueError('Holes cannot themselves contain holes.')

    @cached_property
    def area(self):
        from pyproj import Geod
        geod = Geod(ellps="WGS84")
        area, _ = geod.geometry_area_perimeter(self.to_shapely())
        return area

    def bounding_edges(self, **kwargs) -> List[Tuple[Coordinate, Coordinate]]:
        bounding_coords = self.bounding_coords(**kwargs)
        return list(zip(bounding_coords, [*bounding_coords[1:], bounding_coords[0]]))

    def contains_shape(self, shape: ANY_SHAPE_TYPE, **kwargs) -> bool:
        if isinstance(shape, MultiShapeBase):
            for subshape in shape.geoshapes:
                if not self.contains_shape(subshape):
                    return False
            return True

        if isinstance(shape, PointLike):
            return self.contains_coordinate(shape.centroid)

        s_edges = self.edges(**kwargs)
        o_edges = shape.edges(**kwargs) if isinstance(shape, ShapeLike) else [cast(LineLike, shape).segments]
        if do_edges_intersect(
            [x for edge_ring in s_edges for x in edge_ring],
            [x for edge_ring in o_edges for x in edge_ring]
        ):
            # At least one edge pair intersects - cannot be contained
            return False

        # No edges intersect, so make sure one point along the boundary is
        # contained
        return o_edges[0][0][0] in self

    def edges(self, **kwargs) -> List[List[Tuple[Coordinate, Coordinate]]]:
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
        rings = self.linear_rings(**kwargs)
        return [
            list(zip(ring, ring[1:]))
            for ring in rings
        ]

    def intersects_shape(self, shape: ANY_SHAPE_TYPE, **kwargs) -> bool:
        if isinstance(shape, MultiShapeBase):
            for subshape in shape.geoshapes:
                if self.intersects_shape(subshape, **kwargs):
                    return True

            return False

        if isinstance(shape, PointLike):
            return shape in self

        s_edges = self.edges(**kwargs)
        o_edges = shape.edges(**kwargs) if isinstance(shape, ShapeLike) else [cast(LineLike, shape).segments]
        if do_edges_intersect(
            [x for edge_ring in s_edges for x in edge_ring],
            [x for edge_ring in o_edges for x in edge_ring]
        ):
            # At least one edge pair intersects
            return True

        # If no edges intersect, one shape could still contain the other
        # which counts as intersection. Have to use a point from the boundary
        # because the centroid may fall in a hole
        return o_edges[0][0][0] in self or s_edges[0][0][0] in shape

    def linear_rings(self, **kwargs) -> List[List[Coordinate]]:
        return [
            self.bounding_coords(**kwargs),
            *[list(reversed(shape.bounding_coords())) for shape in self.holes]
        ]

    def to_geojson(
        self,
        properties: Optional[Dict] = None,
        **kwargs
    ) -> Dict:
        k = kwargs.pop('k', None)
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
                for ring in self.linear_rings()
            ]
        )

    def _to_shapely(self):
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

    def to_wkt(self, **kwargs) -> str:
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


class GeoPolygon(ShapeBase):

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
        holes: Optional[Sequence[ShapeLike]] = None,
        dt: Optional[GEOTIME_TYPE] = None,
        properties: Optional[Dict] = None,
        _is_hole: bool = False,
    ):
        super().__init__(holes=holes, dt=dt, properties=properties)

        if not outline[0] == outline[-1]:
            LOGGER.warning(
                'Polygon outlines must be self-closing; your final point will be '
                'connected to your starting point.'
            )
            outline = [*outline, outline[0]]

        if not is_counter_clockwise(outline) ^ _is_hole:
            warn_once(
                'Polygon violates the right hand rule. Inverting coordinate '
                'order; this warning will not repeat.'
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

    @cached_property
    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        lons, lats = cast(
            Tuple[List[float], List[float]],
            zip(*[y.to_float() for y in self.outline])
        )
        return (min(lons), max(lons)), (min(lats), max(lats))

    @cached_property
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
        for edge in zip(polygon, [*polygon[1:], polygon[0]]):
            intersection = find_line_intersection(test_line, edge)
            if not intersection:
                continue

            if intersection[1] and not include_boundary:
                # Lies on boundary, no need to continue
                return False

            if include_boundary or not intersection[1]:
                # If boundaries are allowed, or is not a boundary intersection
                _intersections += 1

        return _intersections > 0 and _intersections % 2 != 0

    def bounding_coords(self, **kwargs) -> List[Coordinate]:
        return self.outline

    def circumscribing_circle(self) -> 'GeoCircle':
        ctr, rad = circumscribing_circle_for_polygon(self.outline[:-1], [])
        return GeoCircle(cast(Coordinate, ctr), cast(float, rad), dt=self.dt)

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
        holes: List[ShapeLike] = []
        if len(rings) > 1:
            holes = [GeoPolygon(ring) for ring in rings[1:]]

        properties = gjson.get('properties', {})
        dt = get_dt_from_geojson_props(
            properties,
            time_start_property,
            time_end_property,
            time_format
        )

        return GeoPolygon(rings[0], holes=holes, dt=dt, properties=properties)

    @classmethod
    def from_pyshp(cls, shape, dt: Optional[GEOTIME_TYPE] = None, properties: Optional[Dict] = None):
        """
        Create a GeoPolygon from a pyshyp polygon.

        Args:
            shape:
                A polygon from the pyshp library

            dt:
                Optional time bounds (presented as a datetime or geostructures.TimeInterval)

            properties:
                Optional additional properties to associate to the shape

        Returns:
            GeoPolygon
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

        gp = GeoPolygon.from_geojson(
            {
                'type': 'Feature',
                'geometry': shape.__geo_interface__,
                'properties': properties
            }
        )
        gp.set_dt(dt)
        return gp

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
        dt: Optional[GEOTIME_TYPE] = None,
        properties: Optional[Dict] = None
    ) -> 'GeoPolygon':
        """Create a GeoPolygon from a wkt string"""
        if not _RE_POLYGON_WKT.match(wkt_str):
            raise ValueError(f'Invalid WKT Polygon: {wkt_str}')

        coord_groups = _RE_LINEAR_RING.findall(wkt_str)
        outline = parse_wkt_linear_ring(coord_groups[0])
        holes: List[ShapeLike] = []
        if len(coord_groups) > 1:
            holes = [
                GeoPolygon(parse_wkt_linear_ring(coord_group))
                for coord_group in coord_groups[1:]
            ]

        return GeoPolygon(outline, holes=holes, dt=dt, properties=properties)

    def to_wkt(self, **kwargs) -> str:
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

    def to_polygon(self, **_):
        return self


class GeoBox(ShapeBase):

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
        holes: Optional[List[ShapeLike]] = None,
        dt: Optional[GEOTIME_TYPE] = None,
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
    def centroid(self) -> Coordinate:
        _nw = self.nw_bound.to_float()
        _se = self.se_bound.to_float()
        return Coordinate(
            round_half_up(statistics.mean([_nw[0], _se[0]]), 7),
            round_half_up(statistics.mean([_nw[1], _se[1]]), 7),
        )

    def bounding_coords(self, **kwargs) -> List[Coordinate]:
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

    def circumscribing_rectangle(self) -> 'GeoBox':
        return self

    def circumscribing_circle(self) -> 'GeoCircle':
        return GeoCircle(
            self.centroid,
            haversine_distance_meters(self.nw_bound, self.centroid),
            dt=self.dt,
        )

    def contains_coordinate(self, coord: Coordinate) -> bool:
        if not (
            self.nw_bound.longitude <= coord.longitude <= self.se_bound.longitude and
            self.se_bound.latitude <= coord.latitude <= self.nw_bound.latitude
        ):
            return False

        for hole in self.holes:
            if coord in hole:
                return False

        return True

    def copy(self) -> 'GeoBox':
        return GeoBox(
            self.nw_bound,
            self.se_bound,
            holes=self.holes.copy(),
            dt=self.dt.copy() if self.dt else None,
            properties=copy.deepcopy(self._properties)
        )

    def to_polygon(self, **kwargs) -> GeoPolygon:
        outer_bound = self.bounding_coords(**kwargs)
        return GeoPolygon(outer_bound, holes=self.holes, dt=self.dt)


class GeoCircle(ShapeBase):

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
        holes: Optional[List[ShapeLike]] = None,
        dt: Optional[GEOTIME_TYPE] = None,
        properties: Optional[Dict] = None,
    ):
        super().__init__(holes=holes, dt=dt, properties=properties)
        self.center = center
        self.radius = radius

    def __eq__(self, other) -> bool:
        if not isinstance(other, GeoCircle):
            return False

        return (
            self.center == other.center
            and self.radius == other.radius
            and self.dt == other.dt
        )

    def __hash__(self) -> int:
        return hash((self.centroid, self.radius, self.dt))

    def __repr__(self) -> str:
        return f'<GeoCircle at {self.centroid.to_float()}; radius {self.radius} meters>'

    @cached_property
    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        nw_bound = inverse_haversine_degrees(
            self.center, 315, self.radius * math.sqrt(2)
        )
        se_bound = inverse_haversine_degrees(
            self.center, 135, self.radius * math.sqrt(2)
        )
        return (nw_bound.longitude, se_bound.longitude), (se_bound.latitude, nw_bound.latitude)

    @property
    def centroid(self) -> Coordinate:
        return self.center

    def bounding_coords(self, **kwargs) -> List[Coordinate]:
        k = kwargs.get('k') or 36
        coords = []

        for i in range(k, -1, -1):
            angle = math.pi * 2 / k * i
            coord = inverse_haversine_radians(self.center, angle, self.radius)
            coords.append(coord)

        return coords

    def circumscribing_circle(self) -> 'GeoCircle':
        return self

    def contains_coordinate(self, coord: Coordinate) -> bool:
        if not haversine_distance_meters(coord, self.center) <= self.radius:
            return False

        for hole in self.holes:
            if coord in hole:
                return False

        return True

    def copy(self) -> 'GeoCircle':
        return GeoCircle(
            self.center,
            self.radius,
            holes=self.holes.copy(),
            dt=self.dt.copy() if self.dt else None,
            properties=copy.deepcopy(self._properties)
        )

    def to_polygon(self, **kwargs) -> GeoPolygon:
        return GeoPolygon(self.bounding_coords(**kwargs), holes=self.holes, dt=self.dt)


class GeoEllipse(ShapeBase):

    """
    An ellipsoid shape (or oval), represented by:
        * a Coordinate center
        * a semi major axis (the radius at its greatest value)
        * a semi minor axis (the radius at its least value)
        * rotation (the major axis's degree offset from North)

    Args:
        center: (Coordinate)
            The centroid of the ellipse

        semi_major: (float)
            The maximum radius value

        semi_minor: (float)
            The minimum radius value

        rotation: (float)
            The major axis's degree offset from North (expressed as East of North)

    """

    def __init__(  # pylint: disable=R0913
        self,
        center: Coordinate,
        semi_major: float,
        semi_minor: float,
        rotation: float,
        holes: Optional[List[ShapeLike]] = None,
        dt: Optional[GEOTIME_TYPE] = None,
        properties: Optional[Dict] = None,
    ):
        super().__init__(holes=holes, dt=dt, properties=properties)

        self.center = center
        self.semi_major = semi_major
        self.semi_minor = semi_minor
        self.rotation = rotation

    def __eq__(self, other) -> bool:
        if not isinstance(other, GeoEllipse):
            return False

        return (
            self.center == other.center
            and self.semi_major == other.semi_major
            and self.semi_minor == other.semi_minor
            and self.rotation == other.rotation
            and self.dt == other.dt
        )

    def __hash__(self) -> int:
        return hash(
            (self.centroid, self.semi_minor, self.semi_major, self.rotation, self.dt)
        )

    def __repr__(self) -> str:
        return (
            f'<GeoEllipse at {self.center.to_float()}; '
            f'radius {self.semi_major}/{self.semi_minor}; '
            f'rotation {self.rotation}>'
        )

    @cached_property
    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        rot_rad = math.radians(self.rotation)
        cos_rot_sq = math.cos(rot_rad)**2
        sin_rot_sq = math.sin(rot_rad)**2
        semi_major_sq = self.semi_major**2
        semi_minor_sq = self.semi_minor**2

        dx = math.sqrt(semi_major_sq * sin_rot_sq + semi_minor_sq * cos_rot_sq)
        dy = math.sqrt(semi_major_sq * cos_rot_sq + semi_minor_sq * sin_rot_sq)

        _, max_lat = inverse_haversine_degrees(self.centroid, 0, dy).to_float()
        max_lon, _ = inverse_haversine_degrees(self.centroid, 90, dx).to_float()
        _, min_lat = inverse_haversine_degrees(self.centroid, 180, dy).to_float()
        min_lon, _ = inverse_haversine_degrees(self.centroid, 270, dx).to_float()

        return (min_lon, max_lon), (min_lat, max_lat)

    @property
    def centroid(self) -> Coordinate:
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
            self.semi_major
            * self.semi_minor
            / math.sqrt(
                (self.semi_major**2) * (math.sin(angle) ** 2)
                + (self.semi_minor**2) * (math.cos(angle) ** 2)
            )
        )

    def bounding_coords(self, **kwargs) -> List[Coordinate]:
        k = kwargs.get('k') or math.ceil(36 * self.semi_major / self.semi_minor)
        coords = []
        rotation = math.radians(self.rotation)

        for i in range(k, -1, -1):
            angle = (math.pi * 2 / k) * i
            radius = self._radius_at_angle(angle)
            coord = inverse_haversine_radians(
                self.center, angle + rotation, radius
            )
            coords.append(coord)

        return coords

    def circumscribing_circle(self) -> GeoCircle:
        return GeoCircle(self.center, self.semi_major, dt=self.dt)

    def contains_coordinate(self, coord: Coordinate) -> bool:
        bearing = bearing_degrees(self.center, coord)
        radius = self._radius_at_angle(math.radians(bearing - self.rotation))
        if not haversine_distance_meters(self.center, coord) <= radius:
            return False

        for hole in self.holes:
            if coord in hole:
                return False

        return True

    def copy(self) -> 'GeoEllipse':
        return GeoEllipse(
            self.center,
            self.semi_major,
            self.semi_minor,
            self.rotation,
            holes=self.holes.copy(),
            dt=self.dt.copy() if self.dt else None,
            properties=copy.deepcopy(self._properties)
        )

    def to_polygon(self, **kwargs) -> GeoPolygon:
        return GeoPolygon(self.bounding_coords(**kwargs), holes=self.holes, dt=self.dt)


class GeoRing(ShapeBase):

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
        holes: Optional[List[ShapeLike]] = None,
        dt: Optional[GEOTIME_TYPE] = None,
        properties: Optional[Dict] = None,
    ):
        super().__init__(holes=holes, dt=dt, properties=properties)
        self.center = center
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.angle_min = angle_min
        self.angle_max = angle_max

    def __eq__(self, other) -> bool:
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

    def __hash__(self) -> int:
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

    def __repr__(self) -> str:
        return (
            f'<GeoRing at {self.center.to_float()}; '
            f'radii {self.inner_radius}/{self.outer_radius}'
            f'{f"; {self.angle_min}-{self.angle_max} degrees" if self.angle_min else ""}>'
        )

    @cached_property
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
    def centroid(self) -> Coordinate:
        if self.angle_min and self.angle_max:
            return self.to_polygon().centroid

        return self.center

    def _draw_bounds(self, **kwargs) -> Tuple[List[Coordinate], List[Coordinate]]:
        """Draws points along the inner and outer radii of the ring/wedge"""
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

    def bounding_coords(self, **kwargs) -> List[Coordinate]:
        outer_bounds, inner_bounds = self._draw_bounds(**kwargs)

        # Is a ring
        if self.angle_min == 0 and self.angle_max == 360:
            return outer_bounds

        # Is a wedge
        return [*outer_bounds, *inner_bounds[::-1], outer_bounds[0]]

    def circumscribing_circle(self) -> GeoCircle:
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

    def copy(self) -> 'GeoRing':
        return GeoRing(
            self.center,
            self.inner_radius,
            self.outer_radius,
            self.angle_min,
            self.angle_max,
            holes=self.holes.copy(),
            dt=self.dt.copy() if self.dt else None,
            properties=copy.deepcopy(self._properties)
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

    def to_polygon(self, **kwargs) -> GeoPolygon:
        rings = self.linear_rings(**kwargs)
        holes = self.holes
        if len(rings) > 1:
            holes += [GeoPolygon(x) for x in rings[1:]]

        return GeoPolygon(
            rings[0],
            holes=holes,
            dt=self.dt,
            properties=self._properties
        )

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


class GeoLineString(BaseShape, LineLike):

    """
    A LineString (or more colloquially, a path) consisting of a series of
    """

    def __init__(
            self,
            vertices: List[Coordinate],
            dt: Optional[GEOTIME_TYPE] = None,
            properties: Optional[Dict] = None
    ):
        super().__init__(dt=dt, properties=properties)
        self.vertices = vertices

    def __eq__(self, other) -> bool:
        if not isinstance(other, GeoLineString):
            return False

        return self.vertices == other.vertices and self.dt == other.dt

    def __hash__(self) -> int:
        return hash((tuple(self.vertices), self.dt))

    def __repr__(self) -> str:
        return f'<GeoLineString with {len(self.vertices)} points>'

    @cached_property
    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        lons, lats = cast(
            Tuple[List[float], List[float]],
            zip(*[y.to_float() for y in self.vertices])
        )
        return (min(lons), max(lons)), (min(lats), max(lats))

    @cached_property
    def centroid(self) -> Coordinate:
        lon, lat = [
            round_half_up(statistics.mean(x), 7)
            for x in zip(*[y.to_float() for y in self.vertices])
        ]
        return Coordinate(lon, lat)

    @property
    def segments(self) -> List[Tuple[Coordinate, Coordinate]]:
        return list(zip(self.vertices, self.vertices[1:]))

    def circumscribing_circle(self) -> GeoCircle:
        centroid = self.centroid
        max_dist = max(haversine_distance_meters(x, centroid) for x in self.vertices)
        return GeoCircle(centroid, max_dist, dt=self.dt)

    def circumscribing_rectangle(self) -> GeoBox:
        lons, lats = zip(*[y.to_float() for y in self.vertices])
        return GeoBox(
            Coordinate(min(lons), max(lats)),
            Coordinate(max(lons), min(lats)),
            dt=self.dt,
        )

    def contains_shape(self, shape: ANY_SHAPE_TYPE, **kwargs) -> bool:
        if isinstance(shape, MultiShapeBase):
            for subshape in shape.geoshapes:
                if not self.contains_shape(subshape):
                    return False
            return True

        if isinstance(shape, ShapeLike):
            return False

        if isinstance(shape, PointLike):
            return self.contains_coordinate(shape.centroid)

        return is_sub_list(cast(LineLike, shape).vertices, self.vertices)

    def contains_coordinate(self, coord: Coordinate) -> bool:
        # For now, just check for exact match. Will need update if buffering
        # becomes a feature
        return coord in self.vertices

    def copy(self) -> 'GeoLineString':
        return GeoLineString(
            self.vertices,
            dt=self.dt.copy() if self.dt else None,
            properties=copy.deepcopy(self._properties)
        )

    @classmethod
    def from_geojson(
        cls,
        gjson: Dict[str, Any],
        time_start_property: str = 'datetime_start',
        time_end_property: str = 'datetime_end',
        time_format: Optional[str] = None,
    ) -> 'GeoLineString':
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
        dt = get_dt_from_geojson_props(
            properties,
            time_start_property,
            time_end_property,
            time_format
        )
        return GeoLineString(coords, dt=dt, properties=properties)

    @classmethod
    def from_pyshp(cls, shape, dt: Optional[GEOTIME_TYPE] = None, properties: Optional[Dict] = None):
        """
        Create a GeoLineString from a pyshyp linestring.

        Args:
            shape:
                A linestring from the pyshp library

            dt:
                Optional time bounds (presented as a datetime or geostructures.TimeInterval)

            properties:
                Optional additional properties to associate to the shape

        Returns:
            GeoLineString
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

        gls = GeoLineString.from_geojson(
            {
                'type': 'Feature',
                'geometry': shape.__geo_interface__,
                'properties': properties
            }
        )
        gls.set_dt(dt)
        return gls

    @classmethod
    def from_shapely(cls, linestring) -> 'GeoLineString':
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
        dt: Optional[GEOTIME_TYPE] = None,
        properties: Optional[Dict] = None
    ) -> 'GeoLineString':
        """Create a GeoLineString from a wkt string"""
        if not _RE_LINESTRING_WKT.match(wkt_str):
            raise ValueError(f'Invalid WKT LineString: {wkt_str}')

        coord_groups = _RE_LINEAR_RING.findall(wkt_str)
        if not len(coord_groups) == 1:
            raise ValueError(f'Invalid WKT LineString: {wkt_str}')

        return GeoLineString(
            parse_wkt_linear_ring(coord_groups[0]),
            dt=dt,
            properties=properties
        )

    def intersects_shape(self, shape: ANY_SHAPE_TYPE, **kwargs) -> bool:
        if isinstance(shape, MultiShapeBase):
            for subshape in shape.geoshapes:
                if self.intersects_shape(subshape, **kwargs):
                    return True

            return False

        if isinstance(shape, PointLike):
            return shape in self

        s_edges = [self.segments]
        o_edges = shape.edges(**kwargs) if isinstance(shape, ShapeLike) else [cast(LineLike, shape).segments]
        if do_edges_intersect(
            [x for edge_ring in s_edges for x in edge_ring],
            [x for edge_ring in o_edges for x in edge_ring]
        ):
            # At least one edge pair intersects
            return True

        # If no edges intersect, one shape could still contain the other
        # which counts as intersection. Have to use a point from the boundary
        # because the centroid may fall in a hole
        return o_edges[0][0][0] in self or s_edges[0][0][0] in shape

    def to_geojson(
            self,
            properties: Optional[Dict] = None,
            **kwargs
    ) -> Dict:
        return {
            'type': 'Feature',
            'geometry': {
                'type': 'LineString',
                'coordinates': [list(x.to_float()) for x in self.vertices],
            },
            'properties': {
                **self._properties_json,
                **(properties or {})
            },
            **kwargs

        }

    def to_polygon(self, **_):
        return GeoPolygon(
            [*self.vertices, self.vertices[0]],
            properties=copy.deepcopy(self._properties),
            dt=self.dt
        )

    def to_pyshp(self, writer):
        return writer.line([[list(x.to_float()) for x in self.vertices]])

    def _to_shapely(self):
        import shapely
        return shapely.LineString([x.to_float() for x in self.vertices])

    def to_wkt(self, **kwargs) -> str:
        bbox_str = self._linear_ring_to_wkt(self.vertices)
        return f'LINESTRING{bbox_str}'


class GeoPoint(BaseShape, PointLike):

    """
    A Coordinate with an associated timestamp. This is the only shape which
    actually requires a timestamp; if your points are time-less you should just
    use the Coordinate object.

    Args:

    """

    def __init__(
        self,
        coordinate: Coordinate,
        dt: Optional[GEOTIME_TYPE] = None,
        properties: Optional[Dict] = None
    ):
        super().__init__(dt=dt, properties=properties)
        self.coordinate = coordinate

    def __eq__(self, other) -> bool:
        if not isinstance(other, GeoPoint):
            return False

        return self.coordinate == other.coordinate and self.dt == other.dt

    def __hash__(self) -> int:
        return hash((self.coordinate, self.dt))

    def __repr__(self) -> str:
        return f'<GeoPoint at {self.coordinate.to_float()}>'

    @property
    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return (
            (self.coordinate.longitude, self.coordinate.longitude),
            (self.coordinate.latitude, self.coordinate.latitude)
        )

    @property
    def centroid(self) -> Coordinate:
        return self.coordinate

    def contains_coordinate(self, coord: Coordinate) -> bool:
        return coord == self.centroid

    def contains_shape(self, shape: ANY_SHAPE_TYPE, **kwargs) -> bool:
        if isinstance(shape, MultiShapeBase):
            for subshape in shape.geoshapes:
                if not self.contains_shape(subshape):
                    return False
            return True

        if isinstance(shape, PointLike):
            return self.contains_coordinate(shape.centroid)

        return False

    def copy(self) -> 'GeoPoint':
        return GeoPoint(
            self.coordinate,
            dt=self.dt.copy() if self.dt else None,
            properties=copy.deepcopy(self._properties)
        )

    def intersects_shape(self, shape: ANY_SHAPE_TYPE, **kwargs) -> bool:
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
    ) -> 'GeoPoint':
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
        dt = get_dt_from_geojson_props(
            properties,
            time_start_property,
            time_end_property,
            time_format
        )

        return GeoPoint(coord, dt=dt, properties=properties)

    @classmethod
    def from_pyshp(cls, shape, dt: Optional[GEOTIME_TYPE] = None, properties: Optional[Dict] = None):
        """
        Create a GeoPoint from a pyshyp point.

        Args:
            shape:
                A point from the pyshp library

            dt:
                Optional time bounds (presented as a datetime or geostructures.TimeInterval)

            properties:
                Optional additional properties to associate to the shape

        Returns:
            GeoPoint
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

        gp = GeoPoint.from_geojson(
            {
                'type': 'Feature',
                'geometry': shape.__geo_interface__,
                'properties': properties
            }
        )
        gp.set_dt(dt)
        return gp

    @classmethod
    def from_shapely(cls, point) -> 'GeoPoint':
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
        dt: Optional[GEOTIME_TYPE] = None,
        properties: Optional[Dict] = None
    ) -> 'GeoPoint':
        """Create a GeoPoint from a wkt string"""
        _match = _RE_POINT_WKT.match(wkt_str)
        if not _match:
            raise ValueError(f'Invalid WKT Point: {wkt_str}')

        coords = _RE_COORD.findall(wkt_str)[0]

        return GeoPoint(
            Coordinate(*coords.split(' ')),
            dt=dt,
            properties=properties
        )

    def to_geojson(
            self,
            properties: Optional[Dict] = None,
            **kwargs
    ) -> Dict:
        return {
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': list(self.coordinate.to_float()),
            },
            'properties': {
                **self._properties_json,
                **(properties or {})
            },
            **kwargs
        }

    def to_pyshp(self, writer):
        return writer.point(*self.centroid.to_float())

    def _to_shapely(self):
        import shapely
        return shapely.Point(self.centroid.longitude, self.centroid.latitude)

    def to_wkt(self, **_) -> str:
        return f'POINT({" ".join(self.coordinate.to_str())})'

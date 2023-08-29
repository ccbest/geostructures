# pylint: disable=C0302
"""
Geospatial shape representations, for use with earth-surface calculations
"""

__all__ = [
    'GeoBox', 'GeoCircle', 'GeoEllipse', 'GeoLineString', 'GeoPoint', 'GeoPolygon',
    'GeoRing', 'GeoShape'
]

from abc import abstractmethod
from datetime import date, datetime
import math
import re
import statistics
from typing import cast, Dict, List, Optional, Union

from geostructures.coordinates import Coordinate
from geostructures.calc import (
    inverse_haversine_radians,
    haversine_distance_meters,
    bearing_degrees,
)
from geostructures.utils.functions import round_half_up
from geostructures.utils.mixins import LoggingMixin, DefaultZuluMixin
from geostructures.time import DateInterval, TimeInterval


_GEOTIME_TYPE = Union[date, datetime, DateInterval, TimeInterval]

_RE_COORD_STR = r'((?:\s?\d+\.?\d*\s\d+\.?\d*\s?\,?)+)'
_RE_COORD = re.compile(_RE_COORD_STR)
_RE_COORD_GROUPS_STR = r'(?:\(' + _RE_COORD_STR + r'\)\,?\s?)+'
_RE_POINT_WKT = re.compile(r'POINT\s?\((\s?\d+\.?\d*\s\d+\.?\d*\s?)\)')
_RE_POLYGON_WKT = re.compile(r'POLYGON\s?\(' + _RE_COORD_GROUPS_STR + r'\)')
_RE_LINESTRING_WKT = re.compile(r'LINESTRING\s?' + _RE_COORD_GROUPS_STR + r'\s?')


def _parse_wkt_coord_group(group: str) -> List[Coordinate]:
    """Parse wkt coordinate list into Coordinate objects"""
    return [
        Coordinate(coord.strip().split(' '))  # type: ignore
        for coord in group.split(',') if coord
    ]


class GeoShape(LoggingMixin, DefaultZuluMixin):

    """Abstract base class for all geoshapes"""

    def __init__(self, dt: Optional[_GEOTIME_TYPE] = None, properties: Optional[Dict] = None):
        super().__init__()
        if isinstance(dt, datetime):
            dt = self._default_to_zulu(dt)

        self.dt = dt
        self.properties = properties or {}

    @abstractmethod
    def __contains__(self, coord: Coordinate):
        """Test whether a coordinate is contained within this geoshape"""

    @abstractmethod
    def __hash__(self) -> int:
        """Create unique hash of this object"""

    @abstractmethod
    def __repr__(self):
        """REPL representation of this object"""

    def _dt_to_json(self) -> Dict[str, str]:
        """"""
        if isinstance(self.dt, datetime):
            return {
                'datetime_start': self.dt.isoformat(),
                'datetime_end': self.dt.isoformat(),
            }

        if isinstance(self.dt, date):
            return {'date_start': self.dt.isoformat(), 'date_end': self.dt.isoformat()}

        if isinstance(self.dt, TimeInterval):
            return {
                'datetime_start': self.dt.start.isoformat(),
                'datetime_end': self.dt.end.isoformat(),
            }

        if isinstance(self.dt, DateInterval):
            return {
                'date_start': self.dt.start.isoformat(),
                'date_end': self.dt.end.isoformat(),
            }

        return {}

    @property
    def start(self) -> Union[date, datetime]:
        """The start date/datetime, if present"""
        if not self.dt:
            raise ValueError("GeoShape has no associated time information.")

        if isinstance(self.dt, (date, datetime)):
            return self.dt

        return self.dt.start

    @property
    def end(self) -> Union[date, datetime]:
        """The end date/datetime, if present"""
        if not self.dt:
            raise ValueError("GeoShape has no associated time information.")

        if isinstance(self.dt, (date, datetime)):
            return self.dt

        return self.dt.end

    @abstractmethod
    def bounding_coords(self, **kwargs) -> List[Coordinate]:
        """
        Produce a list of bounding coordinates for the object. The coordinates will
        necessary not represent smooth curves, therefore some data loss is implied.

        All shapes that represent a linear ring (e.g. a box or polygon) will return
        self-closing coordinates, meaning the last coordinate is equal to the first.

        For shapes with smooth curves (ellipsoids, circles, etc.) you may specify a
        number k that will produce k-points along the curve.

        Keyword Args:
            k: (int)
                For shapes with smooth curves, increasing k increases the number of
                points generated along the curve
        """

    @property
    @abstractmethod
    def centroid(self):
        """
        The center of the shape.

        Returns:
            Coordinate
        """

    @abstractmethod
    def to_polygon(self, **kwargs):
        """
        Converts the shape to a GeoPolygon

        Returns:
            (GeoPolygon)
        """

    @abstractmethod
    def circumscribing_circle(self):
        """
        Produces a circle that entirely encompasses the shape

        Returns:
            (GeoCircle)
        """

    @abstractmethod
    def circumscribing_rectangle(self):
        """
        Produces a rectangle that entirely encompasses the shape

        Returns:
            (GeoBox)
        """

    def to_geojson(
            self,
            k: Optional[int] = None,
            properties: Optional[Dict] = None,
            **kwargs
    ) -> Dict:
        """
        Convert the shape to geojson format.

        Args:
            k: (int)
                For shapes with smooth curves, defines the number of points
                generated along the curve.

            **properties:
                Any number of properties to be included in the geojson properties.

        Returns:
            (dict)
        """
        return {
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[x.to_float() for x in self.bounding_coords(k=k)]],
            },
            'properties': {
                **self.properties,
                **self._dt_to_json(),
                **(properties or {})
            },
            **kwargs
        }

    def to_wkt(self, **kwargs):
        """
        Converts the shape to its WKT string representation

        Keyword Args:
            Arguments to be passed to the .bounding_coords() method. Reference
            that method for a list of corresponding kwargs.

        Returns:
            str
        """
        bbox_str = ",".join(
            " ".join(x.to_str()) for x in self.bounding_coords(**kwargs)
        )
        return f'POLYGON(({bbox_str}))'

    def to_shapely(self):
        """
        Converts the geoshape into a Shapely shape.
        """
        import shapely

        return shapely.geometry.Polygon(
            [[float(x.longitude), float(x.latitude)] for x in self.bounding_coords()]
        )


class GeoPolygon(GeoShape):

    """
    A Polygon, as expressed by an ordered list of Coordinates. The final Coordinate
    must be identical to the first Coordinate.

    Args:
        outline: (List[Coordinate])
            A list of coordinates that define the outside edges of the polygon

        args: (List[Coordinate])
            Additional lists of coordinates representing holes in the polygon

    Keyword Args:
        dt: ()

        properties: dict
            "Storage" space for attributing additional properties to the polygon.

    """

    def __init__(
        self,
        outline: List[Coordinate],
        *args: List[Coordinate],
        dt: Optional[_GEOTIME_TYPE] = None,
        properties: Optional[Dict] = None,
    ):
        super().__init__(dt, properties)

        if not outline[0] == outline[-1]:
            self.logger.warning(
                'Polygon outlines must be self-closing; your final point will be '
                'connected to your starting point.'
            )
            outline.append(outline[0])

        self.outline = outline
        self.holes = args or tuple()

    def __contains__(self, coord: Coordinate) -> bool:
        # First see if the point even falls inside the circumscribing rectangle
        _coord = coord.to_float()
        lons, lats = zip(*[y.to_float() for y in self.outline])
        if (
            min(lons) > _coord[0]
            or min(lats) > _coord[1]
            or max(lons) < _coord[0]
            or max(lats) < _coord[1]
        ):
            # Falls outside rectangle - not in polygon
            return False

        # Use point in polygon to determine if its inside
        return self._point_in_polygon(coord)

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

        s_holes = set([tuple({(x, y) for x, y in zip(hole, hole[1:])}) for hole in self.holes])
        o_holes = set([tuple({(x, y) for x, y in zip(hole, hole[1:])}) for hole in other.holes])
        return s_holes == o_holes

    def __hash__(self):
        return hash((tuple(self.outline), self.dt))

    def __repr__(self):
        return f'<GeoPolygon of {len(self.outline) - 1} coordinates>'

    def _point_in_polygon(self, coord: Coordinate) -> bool:
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

        Returns:
            bool
        """

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        # Create a line from our point that extends eastward past the antimeridian
        # (geo-equivalent of infinity)
        line1 = (coord.to_float(), (181, float(coord.latitude)))
        line1_diff = (line1[0][0] - line1[1][0], line1[0][1] - line1[1][1])

        _intersections = 0
        # zip together consecutive points in the bounding box, representing sides of the polygon
        for line2_coord in zip(self.outline, [*self.outline[1:], self.outline[0]]):
            line2 = (line2_coord[0].to_float(), line2_coord[1].to_float())
            if line2[1][0] < line2[0][0]:
                line2 = (line2[1], line2[0])

            if max(x[0] for x in line2) < line1[0][0]:
                # to the right of line, won't intersect
                continue

            xdiff = (line1_diff[0], line2[0][0] - line2[1][0])
            div = det(xdiff, (line1_diff[1], line2[0][1] - line2[1][1]))
            if div == 0:
                # lines are parallel
                continue

            d = (det(line1[0], line1[1]), det(line2[0], line2[1]))
            x_intersection = det(d, xdiff) / div

            if line1[0][0] == x_intersection:
                # point lies exactly at intersection - counts as being contained
                continue

            if line2[0][0] == x_intersection == line2[1][0]:
                # line is vertical and intersects
                _intersections += 1
                continue

            if (
                line2[0][0] < x_intersection < line2[1][0]
                and line1[0][0] < x_intersection
            ):
                # intersection within x bounds and to the right of the point
                _intersections += 1

        return _intersections > 0 and _intersections % 2 != 0

    def bounding_coords(self, **_):
        # Is self-closing
        return self.outline

    def circumscribing_circle(self):
        centroid = self.centroid
        max_dist = max(
            haversine_distance_meters(x, centroid) for x in self.outline[:-1]
        )
        return GeoCircle(centroid, max_dist, dt=self.dt)

    def circumscribing_rectangle(self):
        lons, lats = zip(*[y.to_float() for y in self.outline])
        return GeoBox(
            Coordinate(min(lons), max(lats)),
            Coordinate(max(lons), min(lats)),
            dt=self.dt,
        )

    @property
    def centroid(self):
        return Coordinate(
            *[
                round_half_up(statistics.mean(x), 7)
                for x in zip(*[y.to_float() for y in self.outline[:-1]])
            ]
        )

    @classmethod
    def from_wkt(cls, wkt_str: str):
        """Create a GeoPolygon from a wkt string"""
        if not _RE_POLYGON_WKT.match(wkt_str):
            raise ValueError(f'Invalid WKT Polygon: {wkt_str}')

        coord_groups = _RE_COORD.findall(wkt_str)
        outline = _parse_wkt_coord_group(coord_groups[0])
        holes = []
        if len(coord_groups) > 1:
            holes = [_parse_wkt_coord_group(coord_group) for coord_group in coord_groups[1:]]

        return GeoPolygon(outline, *holes)

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
        dt: Optional[_GEOTIME_TYPE] = None,
        properties: Optional[Dict] = None,
    ):
        super().__init__(dt, properties)
        self.nw_bound = nw_bound
        self.se_bound = se_bound

    def __contains__(self, coord: Coordinate):
        lon, lat = coord.to_float()
        if float(self.nw_bound.longitude) <= lon <= float(
            self.se_bound.longitude
        ) and float(self.se_bound.latitude) <= lat <= float(self.nw_bound.latitude):
            return True

        return False

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

    def bounding_coords(self, **kwargs):
        _nw = self.nw_bound.to_str()
        _se = self.se_bound.to_str()

        # Is self-closing
        return [
            self.nw_bound,
            Coordinate(_se[0], _nw[1]),
            self.se_bound,
            Coordinate(_nw[0], _se[1]),
            self.nw_bound,
        ]

    def to_polygon(self, **_):
        return GeoPolygon(self.bounding_coords(), dt=self.dt)

    def circumscribing_rectangle(self):
        return self

    def circumscribing_circle(self):
        return GeoCircle(
            self.centroid,
            haversine_distance_meters(self.nw_bound, self.centroid),
            dt=self.dt,
        )

    @property
    def centroid(self):
        _nw = self.nw_bound.to_float()
        _se = self.se_bound.to_float()
        return Coordinate(
            round_half_up(statistics.mean([_nw[0], _se[0]]), 7),
            round_half_up(statistics.mean([_nw[1], _se[1]]), 7),
        )


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
        dt: Optional[_GEOTIME_TYPE] = None,
        properties: Optional[Dict] = None,
    ):
        super().__init__(dt, properties)
        self.center = center
        self.radius = radius

    def __contains__(self, coord: Coordinate):
        return haversine_distance_meters(coord, self.center) <= self.radius

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

    def to_polygon(self, **kwargs):
        return GeoPolygon(self.bounding_coords(**kwargs), dt=self.dt)

    def bounding_coords(self, **kwargs) -> List[Coordinate]:
        k = kwargs.get('k') or 36
        coords = []

        for i in range(k):
            angle = math.pi * 2 / k * i
            coord = inverse_haversine_radians(self.center, angle, self.radius)
            coords.append(coord)

        # Is self-closing
        return [*coords, coords[0]]

    def circumscribing_rectangle(self):
        return GeoBox(
            inverse_haversine_radians(
                self.center, math.radians(315), self.radius * math.sqrt(2)
            ),
            inverse_haversine_radians(
                self.center, math.radians(135), self.radius * math.sqrt(2)
            ),
            dt=self.dt,
        )

    def circumscribing_circle(self):
        return self

    @property
    def centroid(self):
        return self.center


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

        rotation: (int)
            The major axis's degree offset from North (expressed as East of North)

    """

    def __init__(  # pylint: disable=R0913
        self,
        center: Coordinate,
        major_axis: float,
        minor_axis: float,
        rotation: int,
        dt: Optional[_GEOTIME_TYPE] = None,
        properties: Optional[Dict] = None,
    ):
        super().__init__(dt, properties)

        self.center = center
        self.major_axis = major_axis
        self.minor_axis = minor_axis
        self.rotation = rotation

    def __contains__(self, coord: Coordinate):
        bearing = bearing_degrees(self.center, coord)
        radius = self._radius_at_angle(math.radians(bearing - self.rotation))
        return haversine_distance_meters(self.center, coord) <= radius

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

        for i in range(k):
            angle = (math.pi * 2 / k) * i

            radius = self._radius_at_angle(angle)
            coord = inverse_haversine_radians(
                self.center, angle + rotation, radius
            )

            coords.append(coord)

        return [*coords, coords[0]]

    def to_polygon(self, **kwargs):
        return GeoPolygon(self.bounding_coords(**kwargs), dt=self.dt)

    def circumscribing_rectangle(self):
        lons, lats = zip(*[y.to_float() for y in self.bounding_coords()])
        return GeoBox(
            Coordinate(min(lons), max(lats)),
            Coordinate(max(lons), min(lats)),
            dt=self.dt,
        )

    def circumscribing_circle(self):
        return GeoCircle(self.center, self.major_axis, dt=self.dt)

    @property
    def centroid(self):
        return self.center


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

        angle_min: (int) (Optional)
            The minimum angle (expressed as degrees East of North) from which to create a
            wedge shape. If not provided, an angle of 0 degrees will be inferred.

        angle_max: (int) (Optional)
            The maximum angle (expressed as degrees East of North) from which to create a
            wedge shape. If not provided, an angle of 360 degrees will be inferred.

    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        center: Coordinate,
        inner_radius: float,
        outer_radius: float,
        angle_min: Optional[int] = None,
        angle_max: Optional[int] = None,
        dt: Optional[_GEOTIME_TYPE] = None,
        properties: Optional[Dict] = None,
    ):
        super().__init__(dt, properties)

        self.center = center
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.angle_min = angle_min or 0
        self.angle_max = angle_max or 360

    def __contains__(self, coord):
        # Make sure bearing within wedge, if a wedge
        if self.angle_max - self.angle_min < 360:
            bearing = bearing_degrees(self.center, coord)
            if not self.angle_min <= bearing <= self.angle_max:
                return False

        radius = haversine_distance_meters(self.center, coord)
        return self.inner_radius <= radius <= self.outer_radius

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

    def bounding_coords(self, **kwargs):
        k = kwargs.get('k') or max(math.ceil((self.angle_max - self.angle_min) / 10), 10)
        outer_coords = []
        inner_coords = []

        for i in range(k + 1):
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

        # Is self-closing
        return [*outer_coords, *inner_coords[::-1], outer_coords[0]]

    def circumscribing_rectangle(self):
        lons, lats = zip(*[y.to_float() for y in self.bounding_coords()])
        return GeoBox(
            Coordinate(min(lons), max(lats)),
            Coordinate(max(lons), min(lats)),
            dt=self.dt,
        )

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

    @property
    def centroid(self):
        if self.angle_min and self.angle_max:
            # If shape is a wedge, centroid has to shift
            return Coordinate(
                *[
                    round_half_up(statistics.mean(x), 7)
                    for x in zip(*[y.to_float() for y in self.bounding_coords()])
                ]
            )

        return self.center

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

    def to_polygon(self, **kwargs):
        return GeoPolygon(self.bounding_coords(**kwargs), dt=self.dt)


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
        super().__init__(dt, properties)
        self.coords = coords

    def __contains__(self, coord: Coordinate):
        # For now, just check for exact match. Will need update if buffering
        # becomes a feature
        return coord in self.coords

    def __eq__(self, other):
        if not isinstance(other, GeoLineString):
            return False

        return self.coords == other.coords and self.dt == other.dt

    def __hash__(self):
        return hash((tuple(self.coords), self.dt))

    def __repr__(self):
        return f'<GeoLineString with {len(self.coords)} points>'

    def bounding_coords(self, **kwargs):
        # Is not self-closing
        return self.coords

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
                'coordinates': [x.to_float() for x in self.bounding_coords(k=k)],
            },
            'properties': {
                **self.properties,
                **self._dt_to_json(),
                **(properties or {})
            },
            **kwargs
        }

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

    @property
    def centroid(self):
        return Coordinate(
            *[
                round_half_up(statistics.mean(x), 7)
                for x in zip(*[y.to_float() for y in self.coords])
            ]
        )

    @classmethod
    def from_wkt(cls, wkt_str: str):
        """Create a GeoLineString from a wkt string"""
        if not _RE_LINESTRING_WKT.match(wkt_str):
            raise ValueError(f'Invalid WKT LineString: {wkt_str}')

        coord_groups = _RE_COORD.findall(wkt_str)
        if not len(coord_groups) == 1:
            raise ValueError(f'Invalid WKT LineString: {wkt_str}')

        return GeoLineString(_parse_wkt_coord_group(coord_groups[0]))

    def to_wkt(self, **kwargs):
        bbox_str = ",".join(
            " ".join(x.to_str()) for x in self.bounding_coords(**kwargs)
        )
        return f'LINESTRING({bbox_str})'

    def to_polygon(self, **kwargs):
        return GeoPolygon(self.bounding_coords(**kwargs), dt=self.dt)


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
        super().__init__(dt, properties)
        self.center = center

    def __contains__(self, coord: Coordinate):
        # Points don't contain anything, even themselves
        return False

    def __eq__(self, other):
        if not isinstance(other, GeoPoint):
            return False

        return self.center == other.center and self.dt == other.dt

    def __hash__(self):
        return hash((self.center, self.dt))

    def __repr__(self):
        return f'<GeoPoint at {self.center.to_str()}>'

    def bounding_coords(self, **kwargs):
        return [self.center]

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

    def to_shapely(self):
        import shapely

        return shapely.Point(*self.centroid.to_float())

    def circumscribing_circle(self):
        raise NotImplementedError('Points cannot be circumscribed')

    def circumscribing_rectangle(self):
        raise NotImplementedError('Points cannot be circumscribed')

    @property
    def centroid(self):
        return self.center

    @classmethod
    def from_wkt(cls, wkt_str: str):
        """Create a GeoPoint from a wkt string"""
        _match = _RE_POINT_WKT.match(wkt_str)
        if not _match:
            raise ValueError(f'Invalid WKT Point: {wkt_str}')

        return GeoPoint(Coordinate(*_match.groups()[0].split(' ')))

    def to_wkt(self, **_):
        point_str = " ".join(self.center.to_str())
        return f'POINT(({point_str}))'

    def to_polygon(self, **kwargs):
        raise NotImplementedError('Points cannot be converted to polygons')

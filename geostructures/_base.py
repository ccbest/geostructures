"""
Base class declarations for geostructures
"""

from abc import abstractmethod, ABC
from datetime import datetime, timedelta
from functools import lru_cache, cached_property
import re
from typing import (
    Any, Callable, Dict, List, Optional, Protocol, Tuple, TYPE_CHECKING,
    TypeVar, Union, cast
)

from geostructures.coordinates import Coordinate
from geostructures.time import TimeInterval, GEOTIME_TYPE
from geostructures.utils.functions import default_to_zulu, sanitize_json
from geostructures.utils.logging import LOGGER


if TYPE_CHECKING:  # pragma: no cover
    from geostructures import GeoCircle, GeoBox
    from geostructures.typing import GeoShape, PolygonLike


# A wkt coordinate, e.g. '-1.0 2.0', that can be up to 4 numbers long (can include Z and M)
_RE_COORD_STR = r'(?:-?\d{1,3}(?:\.?\d*)?\s?){2,4}\s?'
_RE_COORD = re.compile(_RE_COORD_STR)

# A single linear ring, e.g. '(0.0 0.0, 1.0 1.0, ... )'
_RE_LINEAR_RING_STR = r'\((?:\s?' + _RE_COORD_STR + r'\s?\,?)+\)'
_RE_LINEAR_RING = re.compile(_RE_LINEAR_RING_STR)

# A group of linear rings (shell and holes), e.g. '((0.0 0.0, 1.0 1.0, ... ), ( ... ))'
_RE_LINEAR_RINGS_STR = r'(\((?:' + _RE_LINEAR_RING_STR + r'\,?\s?)+\))'
_RE_LINEAR_RINGS = re.compile(_RE_LINEAR_RINGS_STR)

_RE_ZM_STR = r'\s?([ZM]{0,2})\s?'  # Presence is optional - for matching whole WKT
_RE_ZM = re.compile(r'\s?([ZM]{1,2})\s?')  # Presence is required - for matching ZM specifically

_RE_POINT_WKT = re.compile(r'^POINT' + _RE_ZM_STR + r'\(\s?' + _RE_COORD_STR + r'\s?\)$')
_RE_POLYGON_WKT = re.compile(r'^POLYGON' + _RE_ZM_STR + _RE_LINEAR_RINGS_STR + '$')
_RE_LINESTRING_WKT = re.compile(r'^LINESTRING' + _RE_ZM_STR + _RE_LINEAR_RING_STR + '$')

_RE_MULTIPOINT_WKT = re.compile(r'^MULTIPOINT' + _RE_ZM_STR + _RE_LINEAR_RING_STR + '$')
_RE_MULTIPOLYGON_WKT = re.compile(
    r'^MULTIPOLYGON' + _RE_ZM_STR + r'\((' + _RE_LINEAR_RINGS_STR + r',?\s?)+\)$'
)
_RE_MULTILINESTRING_WKT = re.compile(r'^MULTILINESTRING' + _RE_ZM_STR + _RE_LINEAR_RINGS_STR + '$')


SHAPE_VAR = TypeVar('SHAPE_VAR', bound='BaseShapeProtocol')


class BaseShapeProtocol(Protocol):

    dt: Optional[TimeInterval]
    _properties: Dict
    to_shapely: Callable

    def __contains__(self, other: Union['GeoShape', Coordinate]):
        return self.contains(other)  # pragma: no cover

    @abstractmethod
    def __hash__(self) -> int:
        """Create unique hash of this object"""

    @abstractmethod
    def __repr__(self):
        """REPL representation of this object"""

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
        """The end datetime, if present"""
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
    def _properties_json(self) -> Dict:
        """The shape properties, sanitized to be JSON-serializable"""
        return sanitize_json(self.properties)

    @property
    def start(self) -> datetime:
        """The start date/datetime, if present"""
        if not self.dt:
            raise ValueError("GeoShape has no associated time information.")

        return self.dt.start

    @staticmethod
    def _linear_ring_to_wkt(ring: List[Coordinate]) -> str:
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

    @staticmethod
    def _parse_wkt_linear_ring(wkt_str: str, wkt_coords: str):
        """
        Parses a WKT coordinate string, e.g. "1.0 2.0, 3.0 4.0, ..." into
        geostructures Coordinates. If Z/M values are present, inserts them into
        the dictionary returned in the second element.

        This method assumes the WKT string has already been validated.

        Args:
            wkt_str:
                A WKT geometry, in its entirety.

            wkt_coords:
                A WKT coordinate list

        Returns:
            List of Coordinates
        """
        coords = _RE_COORD.findall(wkt_coords)
        zm = _RE_ZM.findall(wkt_str) or ['ZM']
        parsed_coords = [Coordinate.from_wkt(coord, zm_order=zm[0]) for coord in coords]
        return parsed_coords

    def buffer_dt(
        self: SHAPE_VAR,
        buffer: timedelta,
        inplace: bool = False
    ) -> SHAPE_VAR:
        """
        Adds a timedelta buffer to the beginning and end of dt.

        Args:
            buffer:
                A timedelta that will expand both sides of dt

            inplace:
                Return a new object? Defaults to False.
        """
        if not self.dt:
            raise ValueError("GeoShape has no associated time information.")

        dt = cast(TimeInterval, self.dt)
        shp = self if inplace else self.copy()

        shp.dt = TimeInterval(dt.start-buffer, dt.end+buffer)

        return shp

    def contains(self, shape: Union['GeoShape', Coordinate], **kwargs) -> bool:
        """Test whether a coordinate or GeoShape is contained within this geoshape"""
        if isinstance(shape, Coordinate):
            return self.contains_coordinate(shape)

        if self.dt and shape.dt:
            if not self.contains_time(shape.dt):
                return False

        return self.contains_shape(shape)

    @abstractmethod
    def contains_coordinate(self, coord: Coordinate) -> bool:
        """
        Test if this geoshape contains a coordinate.

        Args:
            coord:
                A Coordinate

        Returns:
            bool
        """

    @abstractmethod
    def contains_shape(self, shape: 'GeoShape', **kwargs) -> bool:
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
    def copy(self: SHAPE_VAR) -> SHAPE_VAR:
        """Produces a copy of the geoshape."""

    def intersects(self, shape: 'GeoShape', **kwargs) -> bool:
        """
        Tests whether another shape intersects this one along both the spatial and time axes.

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
            if not self.intersects_time(shape.dt):
                return False

        return self.intersects_shape(shape, **kwargs)

    @abstractmethod
    def intersects_shape(self, shape: 'GeoShape', **kwargs) -> bool:
        """
        Tests whether another shape intersects this one along its spatial axes.

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

    def intersects_time(self, dt: Union[datetime, TimeInterval]) -> bool:
        """
        Test if the geoshape's time dimension intersects either a point or interval in time.

        Args:
            dt:
                A date or a datetime.

        Returns:
            bool
        """
        if self.dt is None:
            return False

        return self.dt.intersects(dt)

    def set_dt(self, dt: Union[datetime, TimeInterval, None], inplace: bool = False) -> 'BaseShapeProtocol':
        """
        Sets time bounds on this geoshape. Will not mutate unless inplace is True

        Args:
            dt: (Union[datetime, TimeInterval, None])
                A datetime or TimeInterval

            inplace: (bool) (Default False)
                If True, will mutate this object.

        Returns:
            self type
        """
        shape = self if inplace else self.copy()

        if dt is None or isinstance(dt, TimeInterval):
            shape.dt = dt
            return shape

        if isinstance(dt, datetime):
            # Convert to a zero-second time interval
            dt = default_to_zulu(dt)
            shape.dt = TimeInterval(dt, dt)
            return shape

        raise ValueError(f'Unexpected dt value {dt}')

    def set_property(self, key: str, value: Any, inplace: bool = False):
        """
        Sets the value of a property on this geoshape.

        Args:
            key: (str)
                The property name

            value: (Any)
                The property value

            inplace: (bool) (Default False)
                If True, will mutate this object.

        Returns:
            self type
        """
        shape = self if inplace else self.copy()
        shape._properties[key] = value
        return shape

    def strip_dt(self: SHAPE_VAR, inplace: bool = True) -> SHAPE_VAR:
        shape = self if inplace else self.copy()
        shape.dt = None
        return shape

    @abstractmethod
    def to_geojson(
        self,
        properties: Optional[Dict] = None,
        **kwargs
    ) -> Dict:
        """
        Convert the shape to geojson format.

        Args:


            properties: (dict)
                Any number of properties to be included in the geojson properties. These
                values will be unioned with the shape's already defined properties (and
                override them where keys conflict)

        Keyword Args:
            k: (int)
                For shapes with smooth curves, defines the number of points
                generated along the curve.

        Returns:
            (dict)
        """

    @abstractmethod
    def _to_shapely(self):
        """
        Converts the geoshape into a Shapely shape.
        """

    @abstractmethod
    def to_wkt(self, **kwargs) -> str:
        """
        Converts the shape to its WKT string representation

        Keyword Args:
            Arguments to be passed to the .bounding_coords() method. Reference
            that method for a list of corresponding kwargs.

        Returns:
            str
        """


class BaseShape(BaseShapeProtocol, ABC):

    """Abstract base class for all geoshapes"""

    def __init__(
            self,
            dt: Optional[GEOTIME_TYPE] = None,
            properties: Optional[Dict] = None,
            **kwargs
    ):
        super().__init__()
        if kwargs:
            LOGGER.warning('Unrecognized kwargs: %s', ', '.join(kwargs.keys()))
        if isinstance(dt, datetime):
            # Convert to a zero-second time interval
            dt = default_to_zulu(dt)
            self.dt: Optional[TimeInterval] = TimeInterval(dt, dt)
        else:
            self.dt = dt

        self._properties = properties or {}
        self.to_shapely = lru_cache(maxsize=1)(self._to_shapely)


class PolygonLikeMixin(BaseShapeProtocol, ABC):

    """
    Mixin for shapes, singular or multi, that is enclosed by a line, e.g.
    boxes, ellipses, etc.
    """

    holes: List['PolygonLike']

    @property
    @abstractmethod
    def area(self):
        """
        The area of the shape, in meters squared.

        Returns:
            float
        """

    @cached_property
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

    @abstractmethod
    def bounding_coords(self, **kwargs):
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

    @abstractmethod
    def bounding_edges(self, **kwargs):
        """
        Returns a list of edges, defined as a 2-tuple (start and end) of coordinates, that
        represent the polygon's boundary.

        Using discrete coordinates to represent a continuous curve implies some level of data
        loss. You can minimize this loss by specifying k, which represents the number of
        points drawn.

        Does not include information about holes - see the edges method.

        Keyword Args:
            k: (int)
                For shapes with smooth curves, increasing k increases the number of
                points generated along the curve

        Returns:
            A list of 2-tuples representing the edges
        """

    @abstractmethod
    def circumscribing_circle(self) -> 'GeoCircle':
        """
        Produces a circle that entirely encompasses the shape

        Returns:
            (GeoCircle)
        """

    def circumscribing_rectangle(self) -> 'GeoBox':
        """
        Produces a rectangle that entirely encompasses the shape

        Returns:
            (GeoBox)
        """
        from geostructures.structures import GeoBox

        lon_bounds, lat_bounds = self.bounds
        return GeoBox(
            Coordinate(lon_bounds[0], lat_bounds[1]),
            Coordinate(lon_bounds[1], lat_bounds[0]),
            dt=self.dt,
        )

    @abstractmethod
    def edges(self, **kwargs) -> Any:
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

    @abstractmethod
    def linear_rings(self, **kwargs):
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


class LineLikeMixin(BaseShapeProtocol, ABC):

    vertices: List[Coordinate]

    @property
    @abstractmethod
    def segments(self) -> Any:
        """
        Segments (as two-tuples of coordinates) defining the lines between pairs of vertices
        in the linestring(s).

        Returns:
            GeoLineStrings: A list of two-tuples
            MultiGeoLineStrings: A list of the above, one for each sub-linestring
        """

    @abstractmethod
    def circumscribing_circle(self) -> 'GeoCircle':
        """
        Produces a circle that entirely encompasses the shape

        Returns:
            (GeoCircle)
        """

    def circumscribing_rectangle(self) -> 'GeoBox':
        """
        Produces a rectangle that entirely encompasses the shape

        Returns:
            (GeoBox)
        """
        from geostructures.structures import GeoBox

        lon_bounds, lat_bounds = self.bounds
        return GeoBox(
            Coordinate(lon_bounds[0], lat_bounds[1]),
            Coordinate(lon_bounds[1], lat_bounds[0]),
            dt=self.dt,
        )


class PointLikeMixin(BaseShapeProtocol, ABC):
    """
    Class for point-like objects (GeoPoints and MultiPoints).

    There are no defining methods of points; this mixin class merely
    delineates for typing purposes.
    """


class MultiShapeBase(BaseShape, ABC):

    geoshapes: List

    def __eq__(self, other):
        if not isinstance(other, MultiShapeBase):
            return NotImplemented
        return set(self.geoshapes) == set(other.geoshapes) and self.dt == other.dt

    def __hash__(self) -> int:
        return hash((tuple(hash(x) for x in self.geoshapes), self.dt))

    def __iter__(self):
        return self.geoshapes.__iter__()

    @property
    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        min_lons, max_lons, min_lats, max_lats = list(
            zip(*[[x for pair in shape.bounds for x in pair] for shape in self.geoshapes])
        )
        return (min(min_lons), max(max_lons)), (min(min_lats), max(max_lats))

    def contains_coordinate(self, coord: Coordinate) -> bool:
        for shape in self.geoshapes:
            if shape.contains_coordinate(coord):
                return True

        return False

    def contains_shape(self, shape: 'GeoShape', **kwargs) -> bool:
        if isinstance(shape, MultiShapeBase):
            if all(self.contains_shape(subshape, **kwargs) for subshape in shape.geoshapes):
                return True
            return False

        for self_shape in self.geoshapes:
            if self_shape.contains_shape(shape):
                return True

        return False

    def intersects_shape(self, shape: 'GeoShape', **kwargs) -> bool:
        if isinstance(shape, MultiShapeBase):
            for subshape in shape.geoshapes:
                if self.intersects_shape(subshape, **kwargs):
                    return True
            return False

        for self_shape in self.geoshapes:
            if self_shape.intersects_shape(shape):
                return True

            return False

        return False

    def split(self) -> List[BaseShape]:
        shapes = [shape.copy() for shape in self.geoshapes]
        for shape in shapes:
            shape._properties = self._properties.copy()
            shape.dt = self.dt

        return shapes


class SingleShapeBase(BaseShape, ABC):
    pass

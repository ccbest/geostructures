from abc import abstractmethod
from datetime import datetime, timedelta
from functools import lru_cache, cached_property
from typing import Optional, List, Dict, Union, Tuple, cast, Any, TYPE_CHECKING, TypeVar

from geostructures.coordinates import Coordinate
from geostructures.time import TimeInterval
from geostructures.utils.mixins import DefaultZuluMixin


if TYPE_CHECKING:
    from geostructures import GeoCircle, GeoBox


_SHAPE_TYPE = TypeVar('_SHAPE_TYPE', bound='BaseShape')
_GEOTIME_TYPE = Union[datetime, TimeInterval]


class BaseShape(DefaultZuluMixin):

    """Abstract base class for all geoshapes"""

    def __init__(
            self,
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
        self.to_shapely = lru_cache(maxsize=1)(self._to_shapely)

    @abstractmethod
    def __contains__(self, other: Union['BaseShape', Coordinate]):
        """Test whether a coordinate or GeoPoint is contained within this geoshape"""

    @abstractmethod
    def __hash__(self) -> int:
        """Create unique hash of this object"""

    @abstractmethod
    def __repr__(self):
        """REPL representation of this object"""

    @abstractmethod
    def area(self) -> float:
        """
        The area of the shape, in meters squared.

        Returns:
            float
        """

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

    def _dt_to_json(self) -> Dict[str, str]:
        """Safely convert time bounds to json"""
        if not self.dt:
            return {}

        return {
            'datetime_start': self.start.isoformat(),
            'datetime_end': self.end.isoformat(),
        }

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

    @abstractmethod
    def bounding_coords(self, **kwargs) -> Any:
        pass

    @abstractmethod
    def bounding_edges(self, **kwargs) -> List[Tuple[Coordinate, Coordinate]]:
        pass

    def buffer_dt(
        self: _SHAPE_TYPE,
        buffer: timedelta,
        inplace: bool = False
    ) -> _SHAPE_TYPE:
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

    def contains(self, shape: 'BaseShape', **kwargs) -> bool:
        """
        Tests whether this shape fully contains another one, along both
        the spatial and time axes

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
        # TODO: multi shapes
        # Make sure the times overlap, if present on both
        if self.dt and shape.dt:
            if not self.contains_time(shape.dt):
                return False

        return self.contains_shape(shape, **kwargs)

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
    def contains_shape(self, shape: 'BaseShape', **kwargs) -> bool:
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
    def copy(self: _SHAPE_TYPE) -> _SHAPE_TYPE:
        """Produces a copy of the geoshape."""

    @abstractmethod
    def intersects(self, shape: 'BaseShape', **kwargs) -> bool:
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

    @abstractmethod
    def intersects_shape(self, shape: 'BaseShape', **kwargs) -> bool:
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

    @abstractmethod
    def linear_rings(self, **kwargs) -> Any:
        pass

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

    def strip_dt(self: _SHAPE_TYPE) -> _SHAPE_TYPE:
        _copy = self.copy()
        _copy.dt = None
        return _copy

    @abstractmethod
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

    @abstractmethod
    def edges(self, **kwargs) -> Any:
        pass

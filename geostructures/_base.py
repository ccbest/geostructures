
from abc import abstractmethod, ABC
from datetime import datetime
from functools import cached_property, lru_cache
from typing import Any, Dict, List, Optional, Tuple, TypeVar, TYPE_CHECKING, Union

from geostructures.calc import do_edges_intersect
from geostructures.coordinates import Coordinate
from geostructures.utils.mixins import DefaultZuluMixin
from geostructures.time import TimeInterval

if TYPE_CHECKING:
    from geostructures.structures import GeoBox, GeoCircle, GeoPoint, GeoShape


_GEOTIME_TYPE = Union[datetime, TimeInterval]
_SHAPE_TYPE = TypeVar('_SHAPE_TYPE', bound='GeoShape')


class BaseShape(DefaultZuluMixin, ABC):

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
    def __contains__(self, point: Union[Coordinate, 'GeoPoint']):
        """Create unique hash of this object"""

    @abstractmethod
    def __hash__(self) -> int:
        """Create unique hash of this object"""

    @abstractmethod
    def __repr__(self):
        """REPL representation of this object"""

    @abstractmethod
    def area(self):
        """
        The area of the shape, in meters squared.

        Returns:
            float
        """

    @property
    @abstractmethod
    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """q
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

    @abstractmethod
    def volume(self) -> float:
        """
        The volume of the shape, in meters squared seconds.

        Shapes with no time dimension (dt is None), or whose
        time dimension is an instance in time (dt is a datetime)
        will always have a volume of zero.

        Returns:
            float
        """

    def _dt_to_json(self) -> Dict[str, str]:
        """Safely convert time bounds to json"""
        if not self.dt:
            return {}

        return {
            'datetime_start': self.start.isoformat(),
            'datetime_end': self.end.isoformat(),
        }

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

    def bounding_edges(self, **kwargs) -> Union[
        List[Tuple[Coordinate, Coordinate]],
        List[List[Tuple[Coordinate, Coordinate]]]
    ]:
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
            Single shapes: List[Tuple[Coordinate, Coordinate]]
                A list of 2-tuples representing the edges
            Multi shapes: List[List[Tuple[Coordinate, Coordinate]]]
                A list of the single-shape returns, one for each shape in the multi shape
        """

    @abstractmethod
    def circumscribing_circle(self) -> 'GeoCircle':
        """
        Produces a circle that entirely encompasses the shape

        Returns:
            (GeoCircle)
        """

    @abstractmethod
    def circumscribing_rectangle(self) -> 'GeoBox':
        """
        Produces a rectangle that entirely encompasses the shape

        Returns:
            (GeoBox)
        """

    @abstractmethod
    def contains(self, shape: Union['GeoShape', Coordinate], **kwargs) -> bool:
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

    @abstractmethod
    def _contains_coordinate(self, coord: Coordinate) -> bool:
        """
        Test if a geoshape contains a coordinate.

        Args:
            coord:
                A Coordinate

        Returns:
            bool
        """

    @abstractmethod
    def _contains_shape(self, shape: 'GeoShape', **kwargs) -> bool:
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

    def _contains_time(self, dt: Union[datetime, TimeInterval]) -> bool:
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

    def intersects(self, shape: Union['GeoShape', Coordinate], **kwargs) -> bool:
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
        if isinstance(shape, Coordinate):
            return shape in self

        # Make sure the times overlap, if present on both
        if self.dt and shape.dt:
            if not self.intersects_time(shape.dt):
                return False

        return self._intersects_shape(shape, **kwargs)

    def _intersects_shape(self, shape: 'GeoShape', **kwargs) -> bool:
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
        # Make sure the times overlap, if present on both
        if self.dt and shape.dt:
            if not self.intersects_time(shape.dt):
                return False

        if isinstance(shape, GeoPoint):
            return shape in self

        s_edges = self.edges(**kwargs)
        o_edges = shape.edges(**kwargs)
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

    def strip_dt(self: _SHAPE_TYPE) -> _SHAPE_TYPE:
        _copy = self.copy()
        _copy.dt = None
        return _copy

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
    def to_polygon(self, **kwargs) -> 'GeoPolygon':
        """
        Converts the shape to a GeoPolygon

        Returns:
            (GeoPolygon)
        """

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


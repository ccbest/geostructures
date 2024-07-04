"""
Module for geohash transformers
"""


__all__ = [
    'H3Hasher', 'HasherBase', 'NiemeyerHasher',
    'h3_to_geopolygon', 'niemeyer_to_geobox',
]


import abc
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, TypedDict, Union, cast

from geostructures import Coordinate, GeoBox, GeoLineString, GeoPoint, GeoPolygon
from geostructures._base import BaseShape, PointLike, ShapeLike, LineLike, MultiShapeBase
from geostructures._geometry import find_line_intersection
from geostructures.collections import ShapeCollection
from geostructures.multistructures import MultiGeoPoint
from geostructures.time import TimeInterval


_NIEMEYER_CONFIG_TYPE = TypedDict(
    '_NIEMEYER_CONFIG_TYPE',
    {
        'bits': Tuple,
        'charset': str,
        'inverse': Dict[int, int],
        'min_y': float,
        'max_y': float,
        'min_x': float,
        'max_x': float,
    }
)

_NIEMEYER_CONFIG: Dict[int, _NIEMEYER_CONFIG_TYPE] = {
    16: {
        'bits': (8, 4, 2, 1),
        'charset': '0123456789abcdef',
        'inverse': {
            **{x + 48: x for x in range(10)},
            **{x + 97: x + 10 for x in range(6)},
        },
        'min_y': -180.,
        'max_y': 180.,
        'min_x': -180.,
        'max_x': 180.
    },
    32: {
        'bits': (16, 8, 4, 2, 1),
        'charset': '0123456789bcdefghjkmnpqrstuvwxyz',
        'inverse': {
            **{x + 48: x for x in range(10)},
            **{x + 98: x + 10 for x in range(7)},
            **{106: 17, 107: 18, 109: 19, 110: 20},
            **{x + 112: x + 21 for x in range(11)}
        },
        'min_y': -90.,
        'max_y': 90.,
        'min_x': -180.,
        'max_x': 180.
    },
    64: {
        'bits': (32, 16, 8, 4, 2, 1),
        'charset': '0123456789=ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz',
        'inverse': {
            **{x + 48: x for x in range(10)},
            **{x + 65: x + 11 for x in range(26)},
            **{x + 97: x + 38 for x in range(26)},
            **{61: 10, 95: 37}
        },
        'min_y': -180.,
        'max_y': 180.,
        'min_x': -180.,
        'max_x': 180.
    }
}


def _decode_niemeyer(geohash: str, base: int) -> Tuple[float, float, float, float]:
    """
    Converts a Niemeyer geohash into the center lon/lat with corresponding error margins
    Args:
        geohash:
            A geohash

        base:
            The geohash base; one of 16, 32, or 64

    Returns:
        longitude, latitude, longitude_error, latitude_error
    """

    config = _NIEMEYER_CONFIG[base]
    lat_interval = [config['min_y'], config['max_y']]
    lon_interval = [config['min_x'], config['max_x']]
    lon_error, lat_error = config['max_x'], config['max_y']
    lon_component = True

    for character in geohash:
        if character not in config['charset']:
            raise ValueError(f'invalid character in geohash: {character}')

        character_decoded = config['inverse'][ord(character)]
        for mask in config['bits']:
            if lon_component:
                lon_error /= 2.0
                if character_decoded & mask != 0:
                    lon_interval[0] = (lon_interval[0] + lon_interval[1]) / 2.0
                else:
                    lon_interval[1] = (lon_interval[0] + lon_interval[1]) / 2.0
            else:
                lat_error /= 2.0
                if character_decoded & mask != 0:
                    lat_interval[0] = (lat_interval[0] + lat_interval[1]) / 2.0
                else:
                    lat_interval[1] = (lat_interval[0] + lat_interval[1]) / 2.0
            lon_component = not lon_component

    lat = (lat_interval[0] + lat_interval[1]) / 2.0
    lon = (lon_interval[0] + lon_interval[1]) / 2.0

    return lon, lat, lon_error, lat_error


def _coord_to_niemeyer(coordinate: Coordinate, length: int, base: int) -> str:
    """
    Find the geohash (of a specific base/length) in which a lat/lon point falls.

    Args:
        coordinate:
            The coordinate to encode

        length:
            length of geohash

        base:
            the base of the geohash; one of 16, 32, 64

    Return:
        (str) the geohash in which the point falls
    """
    if base not in _NIEMEYER_CONFIG:
        raise ValueError('Unsupported base, must be one of: 16, 32, 64')

    config = _NIEMEYER_CONFIG[base]
    geohash = ''
    lat_interval = [config['min_y'], config['max_y']]
    lon_interval = [config['min_x'], config['max_x']]
    character, bit = 0, 0
    lon_component = True
    lon, lat = coordinate.to_float()

    geohash_position = 0
    while geohash_position < length:
        if lon_component:
            mid = (lon_interval[0] + lon_interval[1]) / 2.0
            if lon > mid:
                character |= config['bits'][bit]
                lon_interval[0] = mid
            else:
                lon_interval[1] = mid
        else:
            mid = (lat_interval[0] + lat_interval[1]) / 2.0
            if lat > mid:
                character |= config['bits'][bit]
                lat_interval[0] = mid
            else:
                lat_interval[1] = mid

        if bit < len(config['bits']) - 1:
            bit += 1
        else:
            geohash += config['charset'][character]
            geohash_position += 1
            character, bit = 0, 0

        lon_component = not lon_component

    return geohash


def _get_niemeyer_subhashes(geohash: str, base: int) -> Set[str]:
    """
    Given a Niemeyer geohash and its base, return the subhashes.

    Args:
        geohash:
            A Niemeyer geohash

        base:
            the base of the geohash; one of 16, 32, 64

    Returns:

    """
    if base not in _NIEMEYER_CONFIG:
        raise ValueError('Unsupported base, must be one of: 16, 32, 64')

    config = _NIEMEYER_CONFIG[base]
    return {geohash + char for char in config['charset']}


def h3_to_geopolygon(
        h3_geohash: str,
        dt: Optional[Union[datetime, TimeInterval]] = None,
        properties: Optional[Dict] = None
) -> GeoPolygon:
    """
    Converts an H3 hashmap into a geostructure FeatureCollection

    Args:
        h3_geohash:
            A H3 geohash, e.g. '88754e6499fffff'

        dt: (Default None)
            The time bound to assign to the GeoPolygon. Use datetime for a time instant
            or TimeInterval (from geostructures.time) for a span of time

        properties: (Default None)
            Any additional properties to assign to the resulting GeoPolygon

    Returns:
        GeoPolygon
    """
    from h3 import h3_to_geo_boundary

    return GeoPolygon(
        [Coordinate(*x) for x in h3_to_geo_boundary(h3_geohash, geo_json=True)],
        dt=dt,
        properties={
            'h3_geohash': h3_geohash,
            **(properties or {})
        }
    )


def niemeyer_to_geobox(
        geohash: str,
        base: int,
        dt: Optional[Union[datetime, TimeInterval]] = None,
        properties: Optional[Dict] = None
) -> GeoBox:
    """
    Convert a Niemeyer geohash to its representative rectangle (a centroid with
    a corresponding error margin).

    Args:
        geohash:
            A Niemeyer geohash

        base:
            the base of the geohash; one of 16, 32, 64

        dt: (Default None)
            The time bound to assign to the GeoPolygon. Use datetime for a time instant
            or TimeInterval (from geostructures.time) for a span of time

        properties: (Default None)
            Any additional properties to assign to the resulting GeoPolygon

    Return:
        GeoBox
    """
    lon, lat, lon_error, lat_error = _decode_niemeyer(geohash, base)
    return GeoBox(
        Coordinate(lon - lon_error, lat + lat_error),
        Coordinate(lon + lon_error, lat - lat_error),
        dt=dt,
        properties={
            'niemeyer_geohash': geohash,
            **(properties or {})
        }
    )


class HasherBase(abc.ABC):

    """
    Base class for all geohasher objects.
    """

    @abc.abstractmethod
    def hash_collection(
        self,
        collection: ShapeCollection,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Returns a dictionary of hashes with values equal to the output of an aggregation
        function over the shapes that intersect each hash.

        Args:
            collection:
                A collection (Track or FeatureCollection) from geostructures.collections

        Keyword Args:
            agg_fn:
                A function that accepts a list of geoshapes. If not specified, the length
                of the list.

        Returns:
            dict, with keys for each geohash and values from agg_fn output
        """

    @abc.abstractmethod
    def hash_shape(
        self,
        shape: BaseShape,
        **kwargs
    ) -> Set[str]:
        """
        Returns a set of the geohashes (as strings) that tile a given geoshape.

        Args:
            shape:
                A geoshape, from geostructures

        Returns:
            set
        """


class H3Hasher(HasherBase):
    """
    Converts geoshapes or collections of geoshapes into H3 geohashes.

    Args:
        resolution:
            The H3 resolution to create geoshapes at. See H3's documentation for additional
            information about resolution sizes.
    """

    def __init__(
        self,
        resolution: Optional[int] = None,
    ):
        import h3  # noqa: F401

        self.resolution = resolution

    @staticmethod
    def _hash_polygon(polygon: BaseShape, resolution: int) -> Set[str]:
        """
        Returns all geohashes contained by a polygon. Uses H3's polyfill function

        Args:
            polygon:
                Any type of geoshape, excluding GeoPoints and GeoLineStrings

            resolution:
                The H3 resolution

        Returns:
            A set of H3 geohashes
        """
        import h3

        return h3.polyfill(
            polygon.to_geojson()['geometry'],
            resolution,
            geo_json_conformant=True  # uses long/lat order
        )

    @staticmethod
    def _hash_linestring(linestring: GeoLineString, resolution: int) -> Set[str]:
        """
        Returns all geohashes that intersect a linestring.

        Because H3 only returns hexes between hex centroids, we create a 1-ring buffer around
        H3's line and test each hex to make sure it intersects a given element of the
        linestring.

        Args:
            linestring:
                A GeoLineString, from geostructures

            resolution:
                The H3 resolution

        Returns:
            A set of H3 geohashes
        """
        import h3

        _hexes = set()
        for segment in linestring.segments:
            # Get h3's straight line hexes
            line_hashes = h3.h3_line(
                h3.geo_to_h3(segment[0].latitude, segment[0].longitude, resolution),
                h3.geo_to_h3(segment[1].latitude, segment[1].longitude, resolution)
            )
            # Add single ring buffer
            all_hexes = set(
                hexid for rings in h3.hex_ranges(line_hashes, 1).values()
                for ring in rings
                for hexid in ring
            )
            # Test which hexes actually intersect the line
            for _hex in all_hexes:
                bounds = [Coordinate(y, x) for x, y in h3.h3_to_geo_boundary(_hex)]
                for hex_edge in zip(bounds, [*bounds[1:], bounds[0]]):
                    if find_line_intersection(segment, hex_edge):
                        _hexes.add(_hex)
                        break

        return _hexes

    @staticmethod
    def _hash_point(point: Coordinate, resolution: int) -> Set[str]:
        """
        Returns the geohash corresponding to a point.

        Args:
            point:
                A geostructures Coordinate

            resolution:
                The H3 resolution

        Returns:
            A set of H3 geohashes
        """
        import h3

        return {
            h3.geo_to_h3(
                point.latitude,
                point.longitude,
                resolution
            )
        }

    def hash_collection(
            self,
            collection: ShapeCollection,
            **kwargs
    ) -> Dict[str, Any]:
        """
        Hash a geostructures FeatureCollection. Returns a dictionary of hashes with
        values equal to the output of an aggregation function over the shapes that
        intersect each hash.

        Args:
            collection:
                A geoshape collection, from geostructures.collections

        Keyword Args:
            resolution:
                The H3 resolution to apply
            agg_fn:
                A function that accepts a list of geoshapes. If not specified, this
                will be the length of the list.

        Returns:
            A dictionary of H3 geohashes mapped to the result of the aggregation
            function
        """
        resolution = kwargs.get('resolution', self.resolution)
        if not resolution:
            raise ValueError('You must pass a H3 resolution.')

        agg_fn = kwargs.get('agg_fn', len)
        hash_dict: Dict[str, List[BaseShape]] = defaultdict(list)
        for shape in collection.geoshapes:
            for hash in self.hash_shape(shape, resolution=resolution):
                hash_dict[hash].append(shape)
        return {h: agg_fn(shape_list) for h, shape_list in hash_dict.items()}

    def hash_coordinates(self, coordinates: Sequence[Coordinate], **kwargs):
        """
        Hashes a collection of coordinates and counts the number
        of times each hash appears.

        Args:
            coordinates:
                A collection of Coordinates, from geostructures

        Keyword Args:
            resolution:
                The H3 resolution to apply
            agg_fn:
                A function that accepts a list of coordinates. If not specified, this
                will be the length of the list.

        Returns:
            A dictionary of H3 geohashes mapped to the result of the aggregation
            function
        """
        resolution = kwargs.get('resolution', self.resolution)
        agg_fn = kwargs.get('agg_fn', len)
        hash_dict: Dict[str, List[Coordinate]] = defaultdict(list)
        for coordinate in coordinates:
            hash_dict[self._hash_point(coordinate, resolution=resolution).pop()].append(coordinate)
        return {h: agg_fn(coord_list) for h, coord_list in hash_dict.items()}

    def hash_shape(self, shape: BaseShape, **kwargs):
        """
        Hashes a singular shape and returns the list of underlying h3 geohashes

        Args:
            shape:
                The shape to be hashed, from geostructures

        Keyword Args:
            resolution:
                The H3 resolution to apply

        Returns:
            The unique list of hashes that comprise the shape
        """
        resolution = kwargs.get('resolution', self.resolution)
        if not resolution:
            raise ValueError('You must pass a H3 resolution.')

        if isinstance(shape, GeoPoint):
            return self._hash_point(shape.centroid, resolution)

        if isinstance(shape, GeoLineString):
            return self._hash_linestring(shape, resolution)

        return self._hash_polygon(shape, resolution)


class NiemeyerHasher(HasherBase):

    """
    Converts geoshapes or collections of geoshapes into Niemeyer geohashes.

    Args:
        length:
            The length of the Niemeyer hashes to return. Longer length geohashes
            equate to smaller geospatial areas.

        base:
            The geohash algorithm base, one of 16, 32, or 64. The base determines the
            number of subdivisions made when traversing from one geohash to its subordinate
            geohashes.
    """

    def __init__(self, length: int, base: int):
        self.length = length
        self.base = base

    @staticmethod
    def _get_surrounding(geohash: str, base: int) -> List[str]:
        """
        Find all geohashes surrounding the geohash with the same base.

        Args:
            geohash: the central geohash
            base: the base of the geohash

        Return:
            (List[str]) surrounding geohashes
        """
        length = len(geohash)
        lon, lat, lon_err, lat_err = _decode_niemeyer(geohash, base)

        return [
            # from directly above, then clockwise
            _coord_to_niemeyer(Coordinate(lon, lat + lat_err * 2), length, base),
            _coord_to_niemeyer(Coordinate(lon + lon_err * 2, lat + lat_err * 2), length, base),
            _coord_to_niemeyer(Coordinate(lon + lon_err * 2, lat), length, base),
            _coord_to_niemeyer(Coordinate(lon + lon_err * 2, lat - lat_err * 2), length, base),
            _coord_to_niemeyer(Coordinate(lon, lat - lat_err * 2), length, base),
            _coord_to_niemeyer(Coordinate(lon - lon_err * 2, lat - lat_err * 2), length, base),
            _coord_to_niemeyer(Coordinate(lon - lon_err * 2, lat), length, base),
            _coord_to_niemeyer(Coordinate(lon - lon_err * 2, lat + lat_err * 2), length, base),
        ]

    def _hash_linestring(
        self,
        linestring: LineLike
    ):
        """
        Find the geohashes that fall along a linestring.

        Args:
            linestring:
                A geostructures.GeoLineString

        Returns:
            A set of geohashes
        """
        if isinstance(linestring, MultiShapeBase):
            return {
                geohash
                for _line in linestring.geoshapes
                for geohash in self._hash_linestring(_line)
            }

        valid, checked, queue = set(), set(), set()
        start = _coord_to_niemeyer(linestring.vertices[0], self.length, self.base)
        queue.add(start)
        valid.add(start)

        while queue:
            gh = queue.pop()
            for near_gh in self._get_surrounding(gh, self.base):
                if near_gh in checked:
                    continue

                checked.add(near_gh)
                if niemeyer_to_geobox(near_gh, self.base).intersects_shape(linestring):
                    valid.add(near_gh)
                    queue.add(near_gh)

        return valid

    def _hash_point(
        self,
        point: PointLike
    ) -> Set[str]:
        """
        Find the geohash that corresponds to a point.

        Args:
            point:
                A geostructures Coordinate

        Returns:
            A set of geohashes
        """
        if isinstance(point, MultiGeoPoint):
            return {
                geohash
                for _point in point.geoshapes
                for geohash in self._hash_point(_point)
            }

        return {_coord_to_niemeyer(point.centroid, self.length, self.base)}

    def _hash_polygon(
        self,
        polygon: ShapeLike
    ) -> Set[str]:
        """
        Find all geohashes that cover the polygon.

        Args:
            polygon: polygon to cover with geohashes

        Returns:
            (Set[str]) the geohashes that cover the polygon
        """
        if isinstance(polygon, MultiShapeBase):
            return {
                geohash
                for _polygon in polygon.geoshapes
                for geohash in self._hash_polygon(_polygon)
            }

        valid, checked, queue = set(), set(), set()
        start = _coord_to_niemeyer(polygon.bounding_coords()[0], self.length, self.base)
        valid.add(start)
        queue.add(start)

        while queue:
            gh = queue.pop()
            for near_gh in self._get_surrounding(gh, self.base):
                if near_gh in checked:
                    continue

                checked.add(near_gh)
                if niemeyer_to_geobox(near_gh, self.base).intersects_shape(polygon):
                    valid.add(near_gh)
                    queue.add(near_gh)

        return valid

    def hash_collection(
        self,
        collection: ShapeCollection,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Hash a geostructures FeatureCollection. Returns a dictionary of hashes with
        values equal to the output of an aggregation function over the shapes that
        intersect each hash.

        Args:
            collection:
                A geostructures FeatureCollection

        Keyword Args:
            agg_fn:
                A function that accepts a list of geoshapes. If not specified, this
                will be the length of the list.

        Returns:
            A dictionary of Niemeyer geohashes mapped to the result of the
            aggregation function
        """
        agg_fn = kwargs.get('agg_fn', len)
        hash_dict: Dict[str, List[BaseShape]] = defaultdict(list)
        for shape in collection.geoshapes:
            for geohash in self.hash_shape(shape):
                hash_dict[geohash].append(shape)
        return {h: agg_fn(shape_list) for h, shape_list in hash_dict.items()}

    def hash_coordinates(self, coordinates: Sequence[Coordinate], **kwargs):
        """
        Hashes a collection of coordinates and counts the number
        of times each hash appears.

        Args:
            coordinates:
                A collection of Coordinates, from geostructures

        Keyword Args:
            agg_fn:
                A function that accepts a list of coordinates. If not specified, this
                will be the length of the list.

        Returns:
            A dictionary of Niemeyer geohashes mapped to the result of the
            aggregation function
        """
        agg_fn = kwargs.get('agg_fn', len)
        hash_dict: Dict[str, List[Coordinate]] = defaultdict(list)
        for coordinate in coordinates:
            hash_dict[_coord_to_niemeyer(coordinate, self.length, self.base)].append(coordinate)
        return {h: agg_fn(coord_list) for h, coord_list in hash_dict.items()}

    def hash_shape(self, shape: BaseShape, **_):
        """
        Converts a geoshape into a set of geohashes that make up the shape.

        Args:
            shape:
                A geostructures shape

        Returns:
            set
        """

        if isinstance(shape, PointLike):
            return self._hash_point(shape)

        if isinstance(shape, LineLike):
            return self._hash_linestring(shape)

        return self._hash_polygon(cast(ShapeLike, shape))

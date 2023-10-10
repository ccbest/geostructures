"""
Module for geohash transformers
"""

__all__ = ['H3Hasher', 'Hasher']

import abc
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Set, Tuple

from geostructures import Coordinate, GeoBox, GeoLineString, GeoPoint
from geostructures.structures import GeoShape
from geostructures.collections import ShapeCollection
from geostructures.calc import find_line_intersection


_niemeyer_config = {
    16: {
        'bits': [8, 4, 2, 1],
        'charset': '0123456789abcdef',
        'inverse': {
            **{x + 48: x for x in range(10)},
            **{x + 97: x + 10 for x in range(6)},
        },
        'min_y': -180,
        'max_y': 180,
        'min_x': -180,
        'max_x': 180
    },
    32: {
        'bits': [16, 8, 4, 2, 1],
        'charset': '0123456789bcdefghjkmnpqrstuvwxyz',
        'inverse': {
            **{x + 48: x for x in range(10)},
            **{x + 98: x + 10 for x in range(7)},
            **{106: 17, 107: 18, 109: 19, 110: 20},
            **{x + 112: x + 21 for x in range(11)}
        },
        'min_y': -90,
        'max_y': 90,
        'min_x': -180,
        'max_x': 180
    },
    64: {
        'bits': [32, 16, 8, 4, 2, 1],
        'charset': '0123456789=ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz',
        'inverse': {
            **{x + 48: x for x in range(10)},
            **{x + 65: x + 11 for x in range(26)},
            **{x + 97: x + 38 for x in range(26)},
            **{61: 10, 95: 37}
        },
        'min_y': -180,
        'max_y': 180,
        'min_x': -180,
        'max_x': 180
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
    if base not in _niemeyer_config:
        raise ValueError('Unsupported base, must be one of: 16, 32, 64')

    config = _niemeyer_config[base]
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


def coord_to_niemeyer(coordinate: Coordinate, length: int, base: int) -> str:
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
    if base not in _niemeyer_config:
        raise ValueError('Unsupported base, must be one of: 16, 32, 64')

    config = _niemeyer_config[base]
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


def get_niemeyer_subhashes(geohash: str, base: int) -> Set[str]:
    """
    Given a Niemeyer geohash and its base, return the subhashes.

    Args:
        geohash:
            A Niemeyer geohash

        base:
            the base of the geohash; one of 16, 32, 64

    Returns:

    """
    if base not in _niemeyer_config:
        raise ValueError('Unsupported base, must be one of: 16, 32, 64')

    config = _niemeyer_config[base]
    return {geohash + char for char in config['charset']}


def niemeyer_to_geobox(geohash: str, base: int) -> GeoBox:
    """
    Convert a Niemeyer geohash to its representative rectangle (a centroid with
    a corresponding error margin).

    Args:
        geohash:
            A Niemeyer geohash

        base:
            the base of the geohash; one of 16, 32, 64

    Return:
        (float) center of geohash latitude
        (float) center of geohash longitude
        (float) height of the geohash in degrees latitude
        (float) width of the geohash in degrees longitude
    """
    lon, lat, lon_error, lat_error = _decode_niemeyer(geohash, base)
    return GeoBox(
        Coordinate(lon - lon_error, lat + lat_error),
        Coordinate(lon + lon_error, lat - lat_error)
    )


class Hasher(abc.ABC):

    """
    Base class for all geohasher objects.
    """

    @abc.abstractmethod
    def hash_collection(
        self,
        collection: ShapeCollection,
        **kwargs
    ) -> Dict[str, float]:
        """
        Returns a dictionary which maps each unique geohash observed over a collection
        of geoshapes mapped to the number of shapes it's been observed in.

        Args:
            collection:
                A collection (Track or FeatureCollection) from geostructures.collections

        Returns:
            dict
        """

    @abc.abstractmethod
    def hash_shape(
        self,
        shape: GeoShape,
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


class H3Hasher(Hasher):
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
    def _hash_polygon(polygon: GeoShape, resolution: int) -> Set[str]:
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
        H3's line and test each hex to make sure it intersects a given vertex of the
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
        for vertex in zip(linestring.bounding_coords(), linestring.bounding_coords()[1:]):
            # Get h3's straight line hexes
            line_hashes = h3.h3_line(
                h3.geo_to_h3(vertex[0].latitude, vertex[0].longitude, resolution),
                h3.geo_to_h3(vertex[1].latitude, vertex[1].longitude, resolution)
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
                    if find_line_intersection(vertex, hex_edge):
                        _hexes.add(_hex)
                        break

        return _hexes

    @staticmethod
    def _hash_point(point: GeoPoint, resolution: int) -> Set[str]:
        """
        Returns the geohash corresponding to a point.

        Args:
            point:
                A GeoPoint

            resolution:
                The H3 resolution

        Returns:
            A set of H3 geohashes
        """
        import h3

        return {
            h3.geo_to_h3(
                point.centroid.latitude,
                point.centroid.longitude,
                resolution
            )
        }

    def hash_collection(
            self,
            collection: ShapeCollection,
            **kwargs
    ) -> Dict[str, float]:
        """
        Hashes a collection of geoshapes and counts the number
        of times each hash appears.

        Args:
            collection:
                A geoshape collection, from geostructures.collections

        Keyword Args:
            resolution:
                The H3 resolution to apply

        Returns:
            A dictionary of H3 geohashes mapped to their corresponding
            counts
        """
        resolution = kwargs.get('resolution', self.resolution)
        if not resolution:
            raise ValueError('You must pass a H3 resolution.')

        out_hexes: Dict[str, float] = defaultdict(lambda: 0)

        for shape in collection:
            _hexes = self.hash_shape(shape, resolution=resolution)
            for _hex in _hexes:
                out_hexes[_hex] += 1

        return dict(out_hexes)

    def hash_shape(self, shape: GeoShape, **kwargs):
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
            return self._hash_point(shape, resolution)

        if isinstance(shape, GeoLineString):
            return self._hash_linestring(shape, resolution)

        return self._hash_polygon(shape, resolution)


class NiemeyerHasher(Hasher):

    """
    # TODO
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
            coord_to_niemeyer(Coordinate(lon, lat + lat_err * 2), length, base),
            coord_to_niemeyer(Coordinate(lon + lon_err * 2, lat + lat_err * 2), length, base),
            coord_to_niemeyer(Coordinate(lon + lon_err * 2, lat), length, base),
            coord_to_niemeyer(Coordinate(lon + lon_err * 2, lat - lat_err * 2), length, base),
            coord_to_niemeyer(Coordinate(lon, lat - lat_err * 2), length, base),
            coord_to_niemeyer(Coordinate(lon - lon_err * 2, lat - lat_err * 2), length, base),
            coord_to_niemeyer(Coordinate(lon - lon_err * 2, lat), length, base),
            coord_to_niemeyer(Coordinate(lon - lon_err * 2, lat + lat_err * 2), length, base),
        ]

    def _hash_linestring(
        self,
        linestring: GeoLineString
    ):
        """
        Find the geohashes that fall along a linestring.

        Args:
            linestring:
                A geostructures.GeoLineString

        Returns:
            A set of geohashes
        """
        valid, checked, queue = set(), set(), set()
        start = coord_to_niemeyer(linestring.coords[0], self.length, self.base)
        queue.add(start)

        while queue:
            gh = queue.pop()
            for near_gh in gh.surrounding():
                if near_gh in checked:
                    continue

                checked.add(near_gh)
                if near_gh.intersects(linestring):
                    valid.add(near_gh)
                    queue.add(near_gh)

        return valid

    def _hash_point(
        self,
        point: GeoPoint
    ) -> Set[str]:
        """
        Find the geohash that corresponds to a point.

        Args:
            point:
                A geostructures.GeoPoint

        Returns:
            A set of geohashes
        """
        return {coord_to_niemeyer(point.centroid, self.length, self.base)}

    def _hash_polygon(
        self,
        polygon: GeoShape
    ) -> Set[str]:
        """
        Find all geohashes that cover the polygon.

        Args:
            polygon: polygon to cover with geohashes

        Returns:
            (Set[str]) the geohashes that cover the polygon
        """
        valid, checked, queue = set(), set(), set()
        start = coord_to_niemeyer(polygon.bounding_coords()[0], self.length, self.base)
        queue.add(start)

        while queue:
            gh = queue.pop()
            for near_gh in self._get_surrounding(gh, self.base):
                if near_gh in checked:
                    continue

                checked.add(near_gh)
                if niemeyer_to_geobox(near_gh, self.base).intersects(polygon):
                    print(near_gh, "intersects")
                    valid.add(near_gh)
                    queue.add(near_gh)
                else:
                    print(near_gh, "does not intersect")

        return valid

    def hash_collection(self, collection: ShapeCollection, **_):
        """
        Hash a geostructures FeatureCollection. Returns a dictionary of hashes and
        the corresponding number of shapes the hash has been observed in.

        Args:
            collection:
                A geostructures FeatureCollection

        Returns:
            dict
        """
        counter = Counter()
        for shape in collection.geoshapes:
            counter.update(self.hash_shape(shape))

        return dict(counter)

    def hash_shape(self, shape: GeoShape, **_):
        """
        Converts a geoshape into a set of geohashes that make up the shape.

        Args:
            shape:
                A geostructures shape

        Returns:
            set
        """
        if isinstance(shape, GeoPoint):
            return self._hash_point(shape)

        if isinstance(shape, GeoLineString):
            return self._hash_linestring(shape)

        return self._hash_polygon(shape)


"""
Module for geohash transformers
"""

__all__ = ['H3Hasher', 'Hasher']

import abc
from collections import defaultdict
from typing import Dict, Optional, Set

import h3

from geostructures import Coordinate, GeoPoint, GeoLineString
from geostructures.structures import GeoShape
from geostructures.collections import ShapeCollection
from geostructures.calc import find_line_intersection


class Hasher(abc.ABC):

    """
    Base class for all geohasher objects.
    """

    @abc.abstractmethod
    def hash_collection(
        self,
        collection: ShapeCollection,
        resolution: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Hashes the coordinates in a track. Remember that unlike Synchronous and Asynchronous
        CoTraveler, time is not relevant to this analytic. As such, the 'track' parameter is
        expecting to just receive a list of coordinates without corresponding timestamps.

        Args:
            collection:
                A list of geostructures

            resolution (int): The geohash resolution that the coordinates should be hashed to. If
                              no value is provided, will default to the `resolution` value passed
                              to `__init__()`. If no value was passed to `__init__()`, the
                              resolution of the least-precise coordinate will be used.

        Returns:
            (Dict[str, float]) A dictionary of geohashes, with corresponding values equal to the
            determined weight of that geohash.
        """


class H3Hasher(Hasher):
    """
    Uses start_value series of coordinates (of various accuracies) to produce start_value "heat map"
    of corresponding H3 Hex ids along with their corresponding counts.

    Coordinates that have start_value lower accuracy than the requested resolution (size of
    the H3 hexes to produce) will be applied entropically over the hexes within
    range of the coordinate.

    For example:
    A coordinate with 2-decimal accuracy (1.11, 2.22) would correspond to start_value resolution
    of 7 (see above conversion dictionary). If the requested resolution is 9, then the
    "weight" added to each 9-resolution hex would be 1 / (9 - 7 + 1) = 1/3. The +1 serves
    further down-weight low-resolution coordinates and avoid 1/1 results in cases where
    the resolution is off by 1.

    """

    def __init__(
        self,
        resolution: Optional[int] = None,
    ):
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
        return h3.polyfill(polygon.to_geojson()['geometry'], resolution)

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
            resolution: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Hashes a collection of geoshapes and counts the number
        of times each hash appears.

        Args:
            collection:
                A geoshape collection, from geostructures.collections

            resolution:
                The H3 resolution to apply

        Returns:
            A dictionary of H3 geohashes mapped to their corresponding
            counts
        """
        resolution = resolution or self.resolution
        if not resolution:
            raise ValueError('You must pass a H3 resolution.')

        out_hexes: Dict[str, float] = defaultdict(lambda: 0)

        for shape in collection:
            _hexes = self.hash_shape(shape, resolution)
            for _hex in _hexes:
                out_hexes[_hex] += 1

        return dict(out_hexes)

    def hash_shape(self, shape: GeoShape, resolution: Optional[int] = None):
        """
        Hashes a singular shape and returns the list of underlying h3 geohashes

        Args:
            shape:
                The shape to be hashed, from geostructures

            resolution:
                The H3 resolution to apply

        Returns:
            The unique list of hashes that comprise the shape
        """
        resolution = resolution or self.resolution
        if not resolution:
            raise ValueError('You must pass a H3 resolution.')

        if isinstance(shape, GeoPoint):
            return self._hash_point(shape, resolution)

        if isinstance(shape, GeoLineString):
            return self._hash_linestring(shape, resolution)

        return self._hash_polygon(shape, resolution)

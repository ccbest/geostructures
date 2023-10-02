"""
Module for geohash transformers
"""

__all__ = ['H3Hasher', 'Hasher']

import abc
from collections import defaultdict
from typing import Dict, Optional, Set

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

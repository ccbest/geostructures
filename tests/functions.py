"""
Shared test helpers.

Two comparison styles exist here, for different purposes:

* ``assert_*_equal`` helpers compare coordinates with an absolute tolerance
  (default 1e-7, ~1.1cm at the equator) and do NOT compare time bounds.
  Use these for serialization round-trips, where float formatting may
  introduce tiny drift.

* ``assert_shape_equivalence`` compares coordinates rounded to a decimal
  precision and DOES compare time bounds. Use this when the expected shape
  is written out longhand at limited precision (e.g. geohash decodings).
"""

from datetime import datetime, timezone
import math
from typing import List

import numpy as np
from pytest import approx

from geostructures import (
    Coordinate, GeoEllipse, GeoLineString, GeoPoint, GeoPolygon, GeoRing,
    MultiGeoPolygon, MultiGeoLineString, MultiGeoPoint,
)
from geostructures._const import EARTH_RADIUS_METERS
from geostructures.geodesic import destination_point
from geostructures.typing import GeoShape, LineLike, MultiShape, PointLike, PolygonLike
from geostructures.utils.functions import round_half_up


default_test_datetime = datetime(1970, 1, 1, 0, 0, tzinfo=timezone.utc)


def assert_coordinates_equal(c1: Coordinate, c2: Coordinate, abs_tol=1e-7):
    """
    Asserts that two coordinates are equal within a specified absolute tolerance.

    Delegates to Coordinate.is_close so the test suite and the library agree on what
    "near enough to be the same point" means; this only adds the failure message.

    Args:
        c1: The first Coordinate
        c2: The second Coordinate
        abs_tol: The absolute tolerance for floating point comparison.
                 Default is 1e-7 (approx 1.1cm at the equator).
    """
    assert c1.is_close(c2, abs_tol=abs_tol), f'{c1!r} is not within {abs_tol} of {c2!r}'


def assert_multishapes_equal(m1: MultiShape, m2: MultiShape):
    assert type(m1) == type(m2), f'{type(m1)} != {type(m2)}'
    assert len(m1.geoshapes) == len(m2.geoshapes), \
        f'{len(m1.geoshapes)} shapes != {len(m2.geoshapes)} shapes'

    if isinstance(m1, MultiGeoPolygon):
        fn = assert_geopolygons_equal
    elif isinstance(m1, MultiGeoLineString):
        fn = assert_geolinestrings_equal
    elif isinstance(m1, MultiGeoPoint):
        fn = assert_geopoints_equal
    else:
        raise TypeError(f'Unrecognized multishape type: {type(m1)}')

    for s1, s2 in zip(m1.geoshapes, m2.geoshapes):
        fn(s1, s2)


def assert_geopolygons_equal(p1: GeoPolygon, p2: GeoPolygon):
    """
    Helper to compare two GeoPolygons using approximate equality for coordinates.
    """
    # 1. Compare Outlines
    assert len(p1.outline) == len(p2.outline)
    for c1, c2 in zip(p1.outline, p2.outline):
        assert_coordinates_equal(c1, c2)

    # 2. Compare Holes
    assert len(p1.holes) == len(p2.holes)
    for h1, h2 in zip(p1.holes, p2.holes):
        # Convert holes to polygons to access their outlines normalized
        assert_geopolygons_equal(h1.to_polygon(), h2.to_polygon())


def assert_geolinestrings_equal(p1: GeoLineString, p2: GeoLineString):
    """
    Helper to compare two GeoLinestring using approximate equality for coordinates.
    """
    assert len(p1.vertices) == len(p2.vertices)
    for c1, c2 in zip(p1.vertices, p2.vertices):
        assert_coordinates_equal(c1, c2)


def assert_geopoints_equal(p1: GeoPoint, p2: GeoPoint):
    """
    Helper to compare two GeoPoints using approximate equality for coordinates.
    """
    assert_coordinates_equal(p1.centroid, p2.centroid)


def _assert_shapelike_equivalence(shape1: PolygonLike, shape2: PolygonLike, precision: int = 7):
    shape1_coords = [
        Coordinate(round_half_up(x.longitude, precision), round_half_up(x.latitude, precision))
        for x in shape1.bounding_coords()
    ]
    shape2_coords = [
        Coordinate(round_half_up(x.longitude, precision), round_half_up(x.latitude, precision))
        for x in shape2.bounding_coords()
    ]
    return shape1_coords == shape2_coords and shape1.dt == shape2.dt


def _assert_linelike_equivalence(shape1: LineLike, shape2: LineLike, precision: int = 7):
    shape1_coords = [
        Coordinate(round_half_up(x.longitude, precision), round_half_up(x.latitude, precision))
        for x in shape1.vertices
    ]
    shape2_coords = [
        Coordinate(round_half_up(x.longitude, precision), round_half_up(x.latitude, precision))
        for x in shape2.vertices
    ]
    return shape1_coords == shape2_coords and shape1.dt == shape2.dt


def _assert_pointlike_equivalence(shape1: PointLike, shape2: PointLike, precision: int = 7):
    shape1_coord = Coordinate(
        round_half_up(shape1.centroid.longitude, precision),
        round_half_up(shape1.centroid.latitude, precision)
    )
    shape2_coord = Coordinate(
        round_half_up(shape2.centroid.longitude, precision),
        round_half_up(shape2.centroid.latitude, precision)
    )
    return shape1_coord == shape2_coord and shape1.dt == shape2.dt


def assert_shape_equivalence(shape1: GeoShape, shape2: GeoShape, precision: int = 7):
    """Asserts that two shapes are equivalent to the given precision."""
    assert type(shape1) == type(shape2), f'{type(shape1)} != {type(shape2)}'

    if isinstance(shape1, MultiShape):
        assert len(shape1.geoshapes) == len(shape2.geoshapes), \
            f'{len(shape1.geoshapes)} shapes != {len(shape2.geoshapes)} shapes'
        for x, y in zip(shape1.geoshapes, shape2.geoshapes):
            assert_shape_equivalence(x, y, precision)
        return

    if isinstance(shape1, PolygonLike):
        assert _assert_shapelike_equivalence(shape1, shape2, precision), \
            f'{shape1} not equivalent to {shape2}'
    elif isinstance(shape1, LineLike):
        assert _assert_linelike_equivalence(shape1, shape2, precision), \
            f'{shape1} not equivalent to {shape2}'
    elif isinstance(shape1, PointLike):
        assert _assert_pointlike_equivalence(shape1, shape2, precision), \
            f'{shape1} not equivalent to {shape2}'
    else:
        raise TypeError(f'Unrecognized shape type {type(shape1)}')


def _assert_parsed_equivalent(original: GeoShape, parsed: GeoShape):
    """
    Asserts that a shape parsed back from a serialized format is geometrically
    equivalent to the original. Shapes without a dedicated parser (e.g.
    GeoCircle, which serializes as a polygon) are compared via their
    polygon form.
    """
    if isinstance(parsed, MultiShape):
        assert_multishapes_equal(original, parsed)
    elif isinstance(parsed, GeoPolygon):
        original = original if isinstance(original, GeoPolygon) else original.to_polygon()
        assert_geopolygons_equal(original, parsed)
    elif isinstance(parsed, GeoLineString):
        assert_geolinestrings_equal(original, parsed)
    elif isinstance(parsed, GeoPoint):
        assert_geopoints_equal(original, parsed)
    else:
        raise TypeError(f'Unrecognized parsed shape type: {type(parsed)}')


def wkt_round_trip(shape: GeoShape):
    """
    Serializes a shape to WKT, parses it back, and asserts geometric
    equivalence. Returns the parsed shape for further assertions.

    WKT carries no time bounds or properties, so those are not compared.
    """
    from geostructures.parsers import parse_wkt

    parsed = parse_wkt(shape.to_wkt())
    _assert_parsed_equivalent(shape, parsed)
    return parsed


def geojson_round_trip(shape: GeoShape):
    """
    Serializes a shape to GeoJSON, parses it back, and asserts geometric
    equivalence plus preservation of time bounds. Returns the parsed shape.
    """
    from geostructures.parsers import parse_geojson

    parsed = parse_geojson(shape.to_geojson())
    _assert_parsed_equivalent(shape, parsed)
    assert parsed.dt == shape.dt
    return parsed


def shapely_round_trip(shape: GeoShape):
    """
    Converts a shape to its shapely equivalent and back, asserting geometric
    equivalence. Returns the parsed shape.

    Shapely geometries carry no time bounds or properties, so those are
    not compared.
    """
    geom = shape.to_shapely()
    conv_map = {
        'Point': GeoPoint,
        'LineString': GeoLineString,
        'Polygon': GeoPolygon,
        'MultiPoint': MultiGeoPoint,
        'MultiLineString': MultiGeoLineString,
        'MultiPolygon': MultiGeoPolygon,
    }
    parsed = conv_map[geom.geom_type].from_shapely(geom)
    _assert_parsed_equivalent(shape, parsed)
    return parsed


def _points_inside(shape, n: int = 2000, seed: int = 0) -> List[Coordinate]:
    """
    Random points inside a GeoCircle, GeoEllipse or GeoRing, placed with
    destination_point at a random bearing and a distance within the shape's
    own limits. A tenth of the ring points sit exactly on a wedge's straight
    edges, which an outline does not sample.
    """
    rng = np.random.default_rng(seed)
    if isinstance(shape, GeoRing):
        bearings = rng.uniform(shape.angle_min, shape.angle_max, n)
        bearings[: n // 20] = shape.angle_min
        bearings[n // 20: n // 10] = shape.angle_max
        distances = rng.uniform(shape.inner_radius, shape.outer_radius, n)
    elif isinstance(shape, GeoEllipse):
        bearings = rng.uniform(0., 360., n)
        limits = [shape._radius_at_angle(math.radians(b - shape.rotation)) for b in bearings]
        distances = rng.uniform(0., 1., n) * np.array(limits)
    else:
        bearings = rng.uniform(0., 360., n)
        distances = rng.uniform(0., shape.radius, n)

    return [
        destination_point(shape.center, float(b), float(d))
        for b, d in zip(bearings, distances)
    ]


def assert_bounds_contain_shape(shape, largest_radius: float):
    """
    Asserts that a curved shape's bounds contain the shape, tightly:

    * every point of a dense outline lies inside bounds (1e-9 degrees)
    * no side of bounds is looser than the outline's extent by more than
      0.01% of the shape's largest radius
    * 2,000 random points inside the shape lie inside bounds

    Longitudes are compared relative to the shape's center so that shapes
    crossing the antimeridian are measured correctly.
    """
    tol = 1e-9
    center_lon = shape.center.longitude

    def rel(lon: float) -> float:
        return (lon - center_lon + 180.) % 360. - 180.

    min_lon, min_lat, max_lon, max_lat = shape.bounds
    min_lon, max_lon = rel(min_lon), rel(max_lon)

    def assert_inside(points: List[Coordinate]):
        for point in points:
            lon = rel(point.longitude)
            assert min_lon - tol <= lon <= max_lon + tol, (shape.bounds, point)
            assert min_lat - tol <= point.latitude <= max_lat + tol, (shape.bounds, point)

    outline = shape.bounding_coords(k=7200)
    if isinstance(shape, GeoRing) and shape.angle_max - shape.angle_min < 360:
        # A wedge's straight edges can reach past its arcs at high latitude
        outline += [
            destination_point(shape.center, angle, float(d))
            for angle in (shape.angle_min, shape.angle_max)
            for d in np.linspace(shape.inner_radius, shape.outer_radius, 400)
        ]
    assert_inside(outline)
    assert_inside(_points_inside(shape))

    # Looseness of each side, in meters
    deg_to_m = math.pi / 180 * EARTH_RADIUS_METERS
    west = min(outline, key=lambda c: rel(c.longitude))
    east = max(outline, key=lambda c: rel(c.longitude))
    slack = (
        (rel(west.longitude) - min_lon) * deg_to_m * math.cos(math.radians(west.latitude)),
        (max_lon - rel(east.longitude)) * deg_to_m * math.cos(math.radians(east.latitude)),
        (min(c.latitude for c in outline) - min_lat) * deg_to_m,
        (max_lat - max(c.latitude for c in outline)) * deg_to_m,
    )
    assert max(slack) <= 1e-4 * largest_radius, (shape.bounds, slack)

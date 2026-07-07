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

from pytest import approx

from geostructures import (
    Coordinate, GeoLineString, GeoPoint, GeoPolygon,
    MultiGeoPolygon, MultiGeoLineString, MultiGeoPoint,
)
from geostructures.typing import GeoShape, LineLike, MultiShape, PointLike, PolygonLike
from geostructures.utils.functions import round_half_up


default_test_datetime = datetime(1970, 1, 1, 0, 0, tzinfo=timezone.utc)


def assert_coordinates_equal(c1: Coordinate, c2: Coordinate, abs_tol=1e-7):
    """
    Asserts that two coordinates are equal within a specified absolute tolerance.

    Args:
        c1: The first Coordinate
        c2: The second Coordinate
        abs_tol: The absolute tolerance for floating point comparison.
                 Default is 1e-7 (approx 1.1cm at the equator).
    """
    # Compare Longitude and Latitude using approximate equality
    try:
        assert c1.longitude == approx(c2.longitude, abs=abs_tol)
        assert c1.latitude == approx(c2.latitude, abs=abs_tol)

        # Strictly compare Z and M (usually integers or explicit floats that shouldn't drift)
        # If these are calculated values, use approx for them as well.
        assert c1.z == c2.z
        assert c1.m == c2.m
    except AssertionError as e:
        print(c1.longitude, c1.latitude)
        print(c2.longitude, c2.latitude)
        raise e


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

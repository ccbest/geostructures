
from zipfile import ZipFile
import pytest
from pytest import approx

from geostructures import Coordinate, GeoLineString, GeoPoint, GeoPolygon, FeatureCollection, \
    MultiGeoPolygon, MultiGeoLineString, MultiGeoPoint
from geostructures.typing import MultiShape


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
    fn = None
    if isinstance(m1, MultiGeoPolygon):
        fn = assert_geopolygons_equal
    elif isinstance(m1, MultiGeoLineString):
        fn = assert_geolinestrings_equal
    elif isinstance(m1, MultiGeoPoint):
        fn = assert_geopoints_equal

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
    Helper to compare two GeoLinestring using approximate equality for coordinates.
    """
    assert_coordinates_equal(p1.centroid, p2.centroid)


@pytest.fixture
def pyshp_round_trip(tmp_path):
    """
    Write a FeatureCollection to a temporary zip, then read it back
    and return the new collection.  Usage:

        new_fc = round_trip(original_fc)
    """
    def _rt(fc, **kwargs):
        zip_path = tmp_path / "test.zip"
        with ZipFile(zip_path, "w") as zf:
            fc.to_shapefile(zf, **kwargs)
        return FeatureCollection.from_shapefile(zip_path)
    return _rt

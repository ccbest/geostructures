
from zipfile import ZipFile
import pytest

from geostructures import Coordinate, FeatureCollection


def assert_coordinates_equal(c1, c2, test_precision=7):
    c1 = Coordinate(round(c1.longitude, test_precision), round(c1.latitude, test_precision))
    c2 = Coordinate(round(c2.longitude, test_precision), round(c2.latitude, test_precision))
    assert c1 == c2


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

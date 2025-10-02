
from zipfile import ZipFile
import pytest

from geostructures import FeatureCollection


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

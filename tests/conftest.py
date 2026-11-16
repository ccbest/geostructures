"""Shared pytest fixtures."""

from zipfile import ZipFile

import pytest

from geostructures import FeatureCollection


@pytest.fixture
def pyshp_round_trip(tmp_path):
    """
    Write a FeatureCollection to a temporary zip, then read it back
    and return the new collection.  Usage:

        new_fc = pyshp_round_trip(original_fc)
    """
    def _rt(fc, **kwargs):
        zip_path = tmp_path / "test.zip"
        with ZipFile(zip_path, "w") as zf:
            fc.to_shapefile(zf, **kwargs)
        return FeatureCollection.from_shapefile(zip_path)
    return _rt


@pytest.fixture
def kml_round_trip(tmp_path):
    """
    Serialize a collection to a temporary .kml/.kmz file, read it back with
    ``parse_kml``, and return the new collection.  Usage:

        new_fc = kml_round_trip(original_fc)            # KML
        new_fc = kml_round_trip(original_fc, '.kmz')    # KMZ
    """
    from geostructures.parsers import parse_kml
    from geostructures.serializers import serialize_kml

    def _rt(fc, suffix=".kml", **kwargs):
        path = tmp_path / f"test{suffix}"
        serialize_kml(fc, path, **kwargs)
        return parse_kml(path)
    return _rt

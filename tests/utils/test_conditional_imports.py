
import importlib

import pytest

from geostructures.utils.conditional_imports import import_optional


def test_import_optional_returns_module():
    import h3
    assert import_optional('h3') is h3


def test_import_optional_unmapped_package():
    with pytest.raises(ImportError) as excinfo:
        import_optional('not_a_real_package_xyz')

    assert 'pip install not_a_real_package_xyz' in str(excinfo.value)


def test_import_optional_names_extra(monkeypatch):
    def _raise(name):
        raise ImportError(name)

    monkeypatch.setattr(importlib, 'import_module', _raise)

    with pytest.raises(ImportError) as excinfo:
        import_optional('fastkml')
    assert 'pip install geostructures[kml]' in str(excinfo.value)

    # Submodules map through their top-level package
    with pytest.raises(ImportError) as excinfo:
        import_optional('fastkml.times')
    assert 'pip install geostructures[kml]' in str(excinfo.value)

    with pytest.raises(ImportError) as excinfo:
        import_optional('shapely')
    assert 'pip install geostructures[shapely]' in str(excinfo.value)

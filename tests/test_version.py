
import importlib
import importlib.metadata

import pytest

from geostructures import _version


def test_read_version_file(monkeypatch):
    # Reads the repo VERSION file when running from a source tree
    assert _version._read_version_file().strip() != ''

    # When no VERSION file can be read, raises with a clear message
    def _boom(self, **kwargs):
        raise OSError('unreadable')

    monkeypatch.setattr(_version.Path, 'read_text', _boom)
    with pytest.raises(FileNotFoundError):
        _version._read_version_file()


def test_version_falls_back_when_metadata_missing(monkeypatch):
    def _raise(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, 'version', _raise)
    try:
        reloaded = importlib.reload(_version)
        assert reloaded.__version__ == _version._read_version_file()
    finally:
        monkeypatch.undo()
        importlib.reload(_version)

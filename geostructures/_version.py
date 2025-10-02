"""
Exposes the version of geostructures
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _read_version_file() -> str | None:
    """
    Fallback when running from a source tree without installed metadata. Tries repo-root
    VERSION first, then a copy inside the package.
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / "VERSION",
        here.with_name("VERSION"),
    ]
    for p in candidates:
        try:
            return p.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    return None


try:
    __version__ = version("geostructures")
except PackageNotFoundError:
    __version__ = _read_version_file()

__all__ = ["__version__"]

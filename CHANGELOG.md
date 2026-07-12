# Changelog

All notable changes to `geostructures` are documented here. This project adheres
to [Semantic Versioning](https://semver.org/).

## [0.14.0] — 2026-07-12

Headline features: **round-trippable KML/KMZ + shapefile I/O**, a new
**serialization module**, **pluggable geodesic algorithms**, and a
**packaging modernization** (setup.py → pyproject.toml). This release also
includes a broad correctness and performance pass across the core geometry,
time, and collection types.

### Features

- **Serialization module (`geostructures.serializers`).** A write-side
  counterpart to `geostructures.parsers`:
  - `serialize_kml` — write a collection or shape to a `.kml` or `.kmz` file
    (format inferred from the extension; `kmz=` override), a stream, or bytes.
  - `serialize_geojson` — serialize to a GeoJSON string (or file), with optional
    `indent` pretty-printing.
  - `serialize_wkt` — serialize a single shape to WKT (or file).
  - `serialize_shapefile` — write a zip-archived ESRI shapefile set to a path,
    stream, or bytes.

  Each `serialize_*` returns the serialized form when no destination is given,
  or writes to a path / file-like when one is.
- **KML/KMZ reading (`parsers.parse_kml`).** Parse a `.kml`/`.kmz` file path,
  raw KML string/bytes, raw KMZ bytes, or an in-memory FastKML object into a
  `FeatureCollection`. KMZ archives are detected by zip magic number and all
  member `.kml` files are combined. Requires the `kml` extra (FastKML).
- **`parsers.parse_shapefile`** — read a zip-archived shapefile set into a
  `FeatureCollection`. `parsers` and `serializers` now cover the same four
  formats: `{geojson, wkt, kml, shapefile}`.
- **Pluggable geodesics (`geostructures.geodesic`).** Haversine, Vincenty, and
  Karney implementations of distance, bearing, and destination-point
  (`haversine_*`, `vincenty_*`, `karney_*`), plus the dispatchers
  `distance_meters`, `bearing_degrees`, `destination_point` and a
  `set_geodesic_algorithm()` switch to select the active algorithm globally.
  (Karney requires the `karney` extra / `geographiclib`.)
- **`GeoLineString.length`** (geodesic length in meters) and
  **`GeoLineString.split_by_length(meters)`** (split a line into segments of a
  maximum length).

### ⚠️ Breaking API Changes

- **`parsers.parse_fastkml` was removed**, superseded by the more general
  `parse_kml` (which also accepts in-memory FastKML objects). Migrate
  `parse_fastkml(obj)` → `parse_kml(obj)`.
- **`Coordinate` is now immutable.** Coordinates use `__slots__` and reject
  attribute mutation (raises `AttributeError`). Invalid inputs now raise plain
  `ValueError`/`TypeError` instead of a pydantic `ValidationError` — update any
  `except ValidationError` handlers around coordinate construction.
- **Optional dependencies are no longer auto-installed.** The previous
  `sys.meta_path` import hook / auto-download behavior was removed. Optional
  features now raise a clear `ImportError` naming the extra to install (e.g.
  `pip install geostructures[kml]`). Extras: `kml`, `shapefile`, `df`, `shapely`,
  `proj`, `h3`, `mgrs`, `karney`, plus `all` and `dev`.
- **`Track.from_shapely` now raises** — shapely geometries carry no temporal
  information, which a `Track` requires.
- **`Track` distance calculations now honor the selected geodesic algorithm.**
  Results change if you call `set_geodesic_algorithm('vincenty'|'karney')`
  (default behavior is unchanged).
- **`TimeInterval` is consistently half-open `[start, end)`.** `isdisjoint` /
  `intersection` follow right-open semantics; instants intersect only when
  contained or equal — this can change boundary-touching results.
- **`conversion` now raises on unknown units** rather than failing silently; the
  `mi` and `kn` factors were corrected.
- **Antimeridian `GeoBox`es now raise** on invalid corner ordering.

### Performance

- **~2.2× faster `Coordinate` construction** (0.85µs → 0.38µs), via the
  immutability / `__slots__` rework replacing per-instance pydantic validation.

### Packaging & Internal

- **Migrated from `setup.py` to `pyproject.toml`** (setuptools backend); the
  version is sourced from the top-level `VERSION` file. `setup.py`,
  `requirements.txt`, and `requirements-dev.txt` were removed.
- Conditional imports rewritten around an `import_optional()` helper with
  install-hint errors (now also covering shapely and pyshp).
- Numerous correctness fixes across geometry, WKT parsing, equality/hashing,
  time handling, and collections (see the commit history for specifics).

[0.14.0]: https://github.com/ccbest/geostructures/compare/v0.13.2...v0.14.0

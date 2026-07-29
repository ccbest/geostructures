# Changelog

All notable changes to `geostructures` are documented here. This project adheres
to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- **Zero-area polygons no longer crash `centroid`.** `GeoPolygon.centroid` and
  `MultiGeoPolygon.centroid` raised `ZeroDivisionError` ("weights sum to zero")
  for degenerate rings — collinear or duplicate coordinates — because there was
  no area to weight the triangle decomposition by. Such rings now fall back to
  the mean of their vertices.
- **`GeoPolygon.centroid` now reports a Z value whenever the polygon has one.**
  `has_z` is true if *any* coordinate carries a Z value, but the centroid took
  the first coordinate's Z, yielding `centroid.z is None` on a `has_z` polygon
  whose Z values started later in the outline. The first Z value present is now
  used.

- **`circumscribing_circle()` no longer raises `RecursionError` on large
  polygons.** The minimum bounding circle used a recursive formulation of
  Welzl's algorithm that recursed once per coordinate, so any `GeoPolygon` with
  roughly 990 or more vertices — an ordinary imported coastline or admin
  boundary — hit Python's recursion limit. It is now iterative and has been
  exercised to 100,000 coordinates.
- **`GeoBox.circumscribing_circle()` now contains the box.** It was centered on
  the box's centroid and measured out to a single corner, which at higher
  latitudes left the other corners outside the "circumscribing" circle — by 17%
  of the radius for a box spanning 70–85° N.
- **Bounding circles no longer disturb the caller's random state.** The
  algorithm drew from the global `random` module, so calling
  `circumscribing_circle()` shifted the sequence returned by the caller's own
  `random` calls. It now uses a private generator, seeded per call, which also
  makes results reproducible run to run.
- **`dist_xyz_meters` is accurate at short range.** It took `acos` of a dot
  product, which loses nearly all precision for close coordinates — it reported
  about a tenth of a meter between a coordinate and itself. It now recovers the
  angle with `atan2`, which is well-conditioned across the whole range.

### ⚠️ Behavior Changes

- **`circumscribing_circle()` returns the smallest enclosing circle for every
  shape that computes one numerically.** `GeoLineString`, `GeoRing` wedges,
  `GeoBox`, `MultiGeoPoint`, `MultiGeoLineString`, and `MultiGeoPolygon`
  previously centered the circle on the shape's centroid and measured out to the
  farthest coordinate, giving a circle up to several times larger than
  necessary. `GeoCircle`, `GeoEllipse`, and full `GeoRing`s are unchanged —
  their circles are exact analytically, and computing one from discretized
  boundary coordinates would be smaller than the shape itself. Coordinates
  spanning more than a hemisphere cannot be covered by a spherical cap, and fall
  back to an enclosing but non-minimal circle with a warning.
- **`bounding_edges` no longer emits a zero-length closing edge.** Because
  `bounding_coords` is self-closing, the boundary was closed a second time,
  producing a final `(first_coord, first_coord)` edge. An *n*-coordinate ring
  now yields *n - 1* edges rather than *n*. `edges()` was already correct and is
  unchanged.

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
  member `.kml` files are combined. Byte order marks and XML encoding
  declarations are handled regardless of fastkml's XML backend (lxml or
  stdlib). Open-ended KML `TimeSpan`s (only a `<begin>` or only an `<end>`)
  parse as zero-length intervals at the known endpoint. Requires the `kml`
  extra (FastKML).
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

- **`parsers.parse_fastkml` and `parsers.read_kml` were removed**, superseded
  by the more general `parse_kml` (which accepts file paths, raw KML/KMZ
  strings and bytes, and in-memory FastKML objects). Migrate
  `parse_fastkml(obj)` → `parse_kml(obj)` and `read_kml(path)` →
  `parse_kml(path)`.
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
- **WKT output now carries `Z`/`M`/`ZM` dimensionality designators**
  (e.g. `POINT ZM (1 2 3 4)`), and **M values no longer leak into the Z slot**
  of position-based formats: GeoJSON positions and KML coordinates emit
  `(lon, lat[, z])` per their specs, dropping M rather than corrupting it into
  altitude. Shapefile output is unaffected (M round-trips there). If you relied
  on the old undesignated/M-polluted output, update your consumers.
- **KML property values:** `None`-valued properties are omitted from
  ExtendedData instead of round-tripping as the string `'None'`; KML placemarks
  containing a bare `LinearRing` geometry now parse as `GeoPolygon`s instead of
  being skipped.
- **Z/M-carrying shapes no longer crash or corrupt spatial math.** `bounds`,
  `centroid`, and `circumscribing_rectangle` on (multi)linestrings and polygons,
  collection centroids, `Track.convolve_duplicate_timestamps`, and H3 hashing
  previously either raised (`ValueError`/H3 rejection) or leaked M/Z values into
  results (e.g. a MultiGeoPoint centroid whose `z` was the average of the
  members' M values). All such computations now operate on lon/lat only.
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

# Changelog

All notable changes to `geostructures` are documented here. This project adheres
to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### ⚠️ Breaking API Changes

- **The deprecated geodesic functions in `geostructures.calc` have been
  removed.** They were deprecated with a runtime warning in 0.14.0 and carried
  their own copies of the geodesic formulas, which had drifted from the
  maintained implementations in `geostructures.geodesic`.

  | Removed | Replacement |
  | --- | --- |
  | `calc.bearing_degrees` | `geodesic.bearing_degrees` |
  | `calc.haversine_distance_meters` | `geodesic.distance_meters` |
  | `calc.inverse_haversine_degrees` | `geodesic.destination_point` |
  | `calc.inverse_haversine_radians` | `geodesic.destination_point` |

  `calc.rotate_coordinates` is unaffected and stays where it is.

  Two differences to be aware of when migrating:

  - The replacements dispatch on the algorithm selected with
    `set_geodesic_algorithm` (Haversine by default), where the removed
    functions were always spherical Haversine. Call `geodesic.haversine_*`
    directly to pin the old behavior.
  - **`destination_point` does not carry the starting coordinate's Z value
    through to the result, where `inverse_haversine_*` did.** If you relied on
    that, set it on the result yourself. This difference is deliberate: shapes
    with curved boundaries derive their `bounding_coords` from
    `destination_point` around a center, and those coordinates are required to
    stay 2D so that WKT Z/M designators reflect what is actually emitted.

  `calc.bearing_degrees` already produced results identical to
  `geodesic.bearing_degrees` as of the rounding changes below, and
  `haversine_distance_meters` matches `geodesic.haversine_distance` to within
  floating point noise, so neither of those swaps changes values.

- **Shape coordinate containers are tuples.** `GeoPolygon.outline`,
  `GeoLineString.vertices`, the `holes` of every polygon-like shape, and the
  `geoshapes` of every multishape were lists, and editing them in place
  (`shape.outline.append(...)`) silently desynchronized the shape from its
  cached bounds, shapely conversion and hash. They are now tuples. Read access
  is unchanged; build a new shape to change its geometry. Constructors still
  accept lists, and `GeoPolygon.bounding_coords()` still returns a list (now
  a copy, so modifying it no longer corrupts the polygon).

- **Collection `geoshapes` are tuples.** `FeatureCollection.geoshapes` and
  `Track.geoshapes` were lists, and editing them in place left cached values
  (bounds, convex hull, and the new filtering arrays) describing the old
  contents. Build a new collection instead: `collection + shape` and
  `collection + other` combine, `collection.filter(predicate)` and the
  `filter_by_*` methods subset, and slicing (below) trims.
- **Slicing a `FeatureCollection` returns a `FeatureCollection`**, as slicing a
  `Track` already returned a `Track`, rather than a list of shapes. Indexing a
  single position still returns the shape.

### Deprecated

- **Mutating shapes is deprecated; shapes will become immutable.** Immutable
  shapes keep cached values (bounds, area, shapely conversion) and hashes
  correct, and are a prerequisite for planned array-backed collections. During
  the deprecation period both forms of mutation still work, log a warning, and
  discard the shape's cached values:

  - The `inplace` argument of `set_dt`, `set_property`, `buffer_dt` and
    `strip_dt`. It currently defaults to `True`; in a future release it will
    be removed and these methods will always return a new shape. Pass
    `inplace=False` and use the returned shape:
    `shape = shape.set_dt(dt, inplace=False)`. Code that calls these methods
    and ignores the return value will stop having any effect once the
    deprecation completes.
  - Assigning to a shape's attributes after construction (`shape.dt = ...`,
    `circle.radius = ...`). Create a new shape instead.
  - Assigning to a collection's attributes after construction
    (`collection.geoshapes = ...`). Create a new collection instead.

  Collections cache arrays derived from their shapes, so a shape mutated in
  place (via either deprecated route) after being added to a collection may
  not be reflected in that collection's filters.

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

- **`intersects_shape()` and `contains_shape()` no longer raise `KeyError` on
  shapes that traverse a segment twice.** A `GeoLineString` that doubles back
  along its own path, or a polygon with a spike that retraces an edge, produced
  two sweep line segments that compared equal, so only one entered the active
  set and the second attempt to retire it failed. Segments are now identified by
  position rather than by coordinates.
- **Shapes touching at exactly one boundary point now report an intersection
  regardless of argument order.** Where one segment's highest point was another
  segment's lowest, the sweep line could retire the first before ever testing it
  against the second, so `a.intersects_shape(b)` and `b.intersects_shape(a)`
  could disagree. Events at the same position are now ordered starts first.
- **Horizontal edges no longer linger in the sweep line's active set.** They
  generated two start events and no stop event, so once passed they were tested
  against every remaining edge in the shape. Answers were unaffected; the cost
  was wasted work.

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
- **`round_half_up` actually rounds halves up now.** It nudged values by a fixed
  `10 ** -(precision + 12)` before deferring to `round()`. That epsilon falls
  below the float's own spacing as the value grows, so the rounding mode changed
  silently with the magnitude of the input — at the precisions this library uses
  for coordinates, centroids, and geometric predicates it never took effect at
  all, leaving plain banker's rounding behind a function named for the opposite.
  It is now exact, and rounds halves **away from zero** rather than toward
  positive infinity, so `round_half_up(-0.5, 0)` is `-1.0` where it was `-0.0`.
  The previous direction quietly biased western longitudes and southern
  latitudes against their positive counterparts.
- **Boundary intersections are detected when Z values are present.**
  `find_line_intersection` compared a `(lon, lat)` pair against whole
  `to_float()` tuples, which carry Z when set, so a Z-carrying endpoint could
  never register as a boundary intersection.

- **Vincenty no longer falls back to Haversine silently.** When the iteration
  fails to converge — usually near-antipodal coordinates — `vincenty_distance`
  and `vincenty_bearing` return the spherical Haversine result instead. That is
  the right answer to return, but a caller who selected Vincenty for its
  accuracy was never told they had stopped receiving it. Both now warn once.
- **`serialize_kml` emits an XML declaration for non-UTF-8 output.** An XML
  document with no declaration is UTF-8 by definition, so a document encoded as
  anything else was undecodable by any consumer that did not already know which
  encoding had been chosen. UTF-8 output is unchanged.
- **`serialize_kml` infers KMZ from a stream's name.** The `.kmz` extension was
  only honored for a path, so passing an open file object for a `.kmz` path
  silently wrote uncompressed KML into it.
- **`parse_kml` reports an over-long string as a bad path.** A long garbage
  string reached the filesystem existence check and escaped as
  `OSError: [Errno 36] File name too long` rather than the `FileNotFoundError`
  a short garbage string produces. Strings containing null bytes behaved the
  same way.

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
- **Geometric predicates use an explicit tolerance instead of rounding.**
  `find_line_intersection` and the point-in-polygon ray caster snapped values to
  ten decimal places to absorb float error. Rounding is a grid rather than a
  tolerance, so two values the same tiny distance apart compared equal or
  unequal depending on which side of a grid line they happened to fall on. Both
  now compare against `COORDINATE_TOLERANCE` (1e-6 degrees, roughly 11 cm at the
  equator, matching the six decimal places coordinates are normalized to
  elsewhere). Points nearer an edge than that are treated as lying on it.
- **Computed values are no longer quantized unless you ask.** Functions that
  return values now keep their full precision by default; only functions that
  return formatted strings (`Coordinate.to_dms`, `to_qdms`) still round, because
  their output format requires a fixed number of digits.
  - `GeoBox.centroid` and `GeoLineString.centroid` rounded to seven decimal
    places, discarding roughly a centimeter of real precision and disagreeing
    with `GeoPolygon.centroid`, which never rounded.
  - `calc.bearing_degrees`' `precision` keyword now defaults to `None`, meaning
    no rounding, rather than `5`. Its output is now identical to that of its
    replacement, `geodesic.bearing_degrees`, so migrating no longer changes
    results. Pass `precision=5` to restore the old value.
  - `calc.inverse_haversine_radians` and `calc.inverse_haversine_degrees` no
    longer round to seven decimal places.

  - `Coordinate.to_projection` and `Coordinate.from_projection` no longer round
    to six decimal places. That constant was applied in the target CRS's own
    units, so it meant a micrometer in EPSG:3857 but eleven centimeters in a
    degree-based CRS like EPSG:4269. Reprojecting a coordinate with more than
    six decimal places used to discard up to 5.6 cm irrecoverably.

  `Coordinate.from_qdms` still rounds to six decimal places, matching the
  resolution its `DDDMMSSHH` input format can actually carry.

  **If you compared reprojected coordinates with `==`**, that comparison
  succeeded before only because both sides had been quantized onto the same
  6-decimal grid, and it silently failed for any input finer than that. Use the
  new `Coordinate.is_close` instead, or call `round_half_up(value, 6)` yourself
  to opt back into the old precision.

- **The `serialize_*` functions raise `TypeError` on inputs that are not shapes
  or collections.** They previously reached straight for a `to_wkt`/`to_geojson`
  method and surfaced an `AttributeError` naming a method the caller had never
  heard of.
- **`serialize_shapefile` raises `ValueError` on an empty collection.** A
  shapefile's geometry type is fixed by its header and taken from the shapes
  present, so an empty collection produced an archive containing no members at
  all — 22 bytes of end-of-central-directory record, which reads as a corrupt
  zip rather than as an empty result.

### Features

- **Collections filter with numpy.** `filter_by_dt`, `filter_by_intersection`,
  `filter_contained_by`, `filter_contains`, `intersects`, and `Track` time
  slicing now work from arrays derived from the collection's shapes (computed
  on first use, and carried over to filtered results) rather than testing every
  shape. Candidates are first narrowed by bounding envelope and time; GeoPoints
  tested against a `GeoPolygon` or `GeoBox` are then resolved with a vectorized
  point-in-polygon test, and anything else with the existing per-shape methods,
  so results are unchanged. On 300k timed points, repeated filters run 30-145x
  faster than before.
- **`collection + shape`** adds a single shape to a copy of a `FeatureCollection`
  or `Track`, and **`collection.filter(predicate)`** subsets by an arbitrary
  function, returning the same collection type.
- **`HasherBase.hash_gaussian(shape, shape_probability, grid_probability, max_hashes)`**
  — treats a `GeoCircle` or `GeoEllipse` as enclosing `shape_probability` of a
  2-D Gaussian (e.g. `0.95` for an ellipse containing 95% of the probability)
  and returns the most probable geohashes holding at least `grid_probability` of
  its probability, as `{geohash: probability}`. A Gaussian never reaches zero,
  so `grid_probability` must be under 1 to avoid hashing the whole globe. Starts
  at the hasher's resolution and coarsens as needed to return no more than
  `max_hashes` geohashes, falling back to `{'': 1.}` (the whole globe) if even
  the coarsest resolution needs more.
  Supported by `H3Hasher` and `NiemeyerHasher`, which gain `coarsen()`,
  `get_cell_boundary()` and `get_neighbors()` to support it.
- **`H3Hasher` accepts resolution 0.** The base cells were rejected as though no
  resolution had been given.
- **`Coordinate.is_close(other, abs_tol=...)`** — a tolerance-based comparison
  for coordinates that have been through a calculation, a reprojection, or a
  serialization round trip. Longitude and latitude are compared within
  `abs_tol` (defaulting to the same tolerance geometric predicates use to decide
  whether a point lies on an edge), wrapping across the antimeridian; Z and M
  are compared strictly, since a tolerance in degrees says nothing about a value
  in meters. `==` remains exact — approximate closeness is not transitive, so it
  cannot back `__hash__`.
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

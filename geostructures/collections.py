"""
Module for sequences of GeoShapes
"""

__all__ = ['FeatureCollection', 'CollectionBase', 'Track']

from collections import defaultdict, Counter
from datetime import date, datetime, time, timedelta
from functools import cached_property
from pathlib import Path
import tempfile
from typing import Callable, cast, Any, List, Dict, Iterable, Optional, Union, Sequence, Tuple, TypedDict, TypeVar
from zipfile import ZipFile

import numpy as np
from pydantic import validate_call

from geostructures import Coordinate, LOGGER
from geostructures._base import PolygonLikeMixin, PointLikeMixin, LineLikeMixin, MultiShapeBase, BaseShape
from geostructures._geometry import convex_hull
from geostructures.calc import haversine_distance_meters
from geostructures.multistructures import MultiGeoLineString, MultiGeoPoint, MultiGeoPolygon
from geostructures.structures import GeoLineString, GeoPoint, GeoPolygon
from geostructures.time import TimeInterval
from geostructures.utils.functions import default_to_zulu
from geostructures.utils.logging import warn_once


_COL_TYPE = TypeVar('_COL_TYPE', bound='CollectionBase')


class CollectionBase:

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(self, geoshapes: List[BaseShape]):
        super().__init__()
        self.geoshapes = geoshapes

    def __bool__(self):
        return bool(self.geoshapes)

    def __contains__(self, item):
        return item in self.geoshapes

    def __iter__(self):
        """Iterate through the track"""
        return self.geoshapes.__iter__()

    def __len__(self):
        """The track length"""
        return self.geoshapes.__len__()

    @cached_property
    def bounds(self) -> Tuple[float, float, float, float]:
        all_bounds = [x.bounds for x in self.geoshapes]
        return (
            min(x[0] for x in all_bounds),
            min(x[1] for x in all_bounds),
            max(x[2] for x in all_bounds),
            max(x[3] for x in all_bounds)
        )

    @property
    def centroid(self):
        lat, lon = tuple(
            map(np.average, zip(*[shape.centroid.to_float() for shape in self.geoshapes]))
        )
        return Coordinate(lat, lon)

    @cached_property
    def convex_hull(self):
        """Creates a convex hull around the pings"""
        def _get_vertices(shapes):
            vertices = []
            for shape in shapes:
                if isinstance(shape, MultiShapeBase):
                    vertices += [
                        vertex
                        for _shape in shape.geoshapes
                        for vertex in _get_vertices([_shape])
                    ]
                elif isinstance(shape, PointLikeMixin):
                    vertices.append(shape.centroid)
                elif isinstance(shape, LineLikeMixin):
                    vertices += shape.vertices
                elif isinstance(shape, PolygonLikeMixin):
                    vertices += shape.bounding_coords()
            return vertices

        return GeoPolygon(convex_hull(_get_vertices(self.geoshapes)))

    def filter_by_dt(self: _COL_TYPE, dt: Union[datetime, TimeInterval]) -> _COL_TYPE:
        """
        Subsets the tracks pings according to the date object provided.

        Args:
            dt:
                A date object from geostructures.time

        Returns:
            Track
        """
        # Has to be checked before date - datetimes are dates, but dates are not datetimes
        if isinstance(dt, datetime):
            dt = default_to_zulu(dt)
            return type(self)(
                [x for x in self.geoshapes if x.dt is not None and x.dt == TimeInterval(dt, dt)]
            )

        if isinstance(dt, TimeInterval):
            return type(self)(
                [x for x in self.geoshapes if x.dt is not None and dt.intersects(x.dt)]
            )

        raise ValueError(f"Unexpected dt object: {dt}")

    def filter_by_intersection(self: _COL_TYPE, shape: BaseShape) -> _COL_TYPE:
        """
        Filter the shape collection using an intersecting geoshape, which is optionally
        time-bounded.

        Args:
            shape:
                A geoshape

        Returns:
            A shape collection of the same type as the original
        """
        return type(self)([x for x in self.geoshapes if x.intersects(shape)])

    def filter_contained_by(self: _COL_TYPE, shape: BaseShape) -> _COL_TYPE:
        """
        Filter the shape collection using a containing geoshape, which is optionally
        time-bounded.

        Args:
            shape:
                A geoshape

        Returns:
            A shape collection of the same type as the original
        """
        return type(self)([x for x in self.geoshapes if shape.contains(x)])

    def filter_contains(self: _COL_TYPE, shape: BaseShape) -> _COL_TYPE:
        """
        Filter the shape collection using a geoshape each shape must contain,
        which is optionally time-bounded.

        Args:
            shape:
                A geoshape

        Returns:
            A shape collection of the same type as the original
        """
        return type(self)([x for x in self.geoshapes if x.contains(shape)])

    def filter_by_property(self: _COL_TYPE, property: str, func: Callable[[Any], bool]) -> _COL_TYPE:
        """
        Filter based on property and the given function.
        Args:
            property: The property to filter by.
            func: The function to apply to the property value. e.g.
            lambda x: x == 'red'
        Returns:
            A shape collection of the same type as the original
        """
        filtered_shapes = []
        for shape in self.geoshapes:
            if property not in shape.properties:
                raise KeyError(f"Property '{property}' not found in shape properties")
            if func(shape.properties[property]):
                filtered_shapes.append(shape)
        return type(self)(filtered_shapes)

    @classmethod
    def from_fastkml_folder(cls, folder):
        """
        Construct a FeatureCollection from a FastKML Folder. Placemarks in
        the folder will be parsed into their corresponding geostructures.
        """
        from geostructures.parsers import parse_fastkml

        return FeatureCollection(parse_fastkml(folder))

    @classmethod
    def from_geojson(
        cls,
        gjson: Dict[str, Any],
        time_start_property: str = 'datetime_start',
        time_end_property: str = 'datetime_end',
        time_format: Optional[str] = None,
    ):
        """
        Creates a Track or FeatureCollection from a GeoJSON FeatureCollection.

        Args:
            gjson:
                A geojson object (dictionary)

            time_start_property:
                The name of the property describing the start time (if available)

            time_end_property:
                The name of the property describing the end time (if available)

            time_format: (Optional)
                The format of the timestamps in the above time fields.

        Returns:
            Track or FeatureCollection
        """
        from geostructures.parsers import parse_geojson

        if gjson.get('type') != 'FeatureCollection':
            raise ValueError('Malformed GeoJSON; expected FeatureCollection')

        shapes: List[BaseShape] = []
        for feature in gjson.get('features', []):
            shapes.append(
                parse_geojson(
                    feature,
                    time_start_property,
                    time_end_property,
                    time_format
                )
            )

        return cls(shapes)

    @classmethod
    def from_geopandas(
        cls,
        df,
        time_start_field: str = 'datetime_start',
        time_end_field: str = 'datetime_end',
    ):
        """
        Creates a Track or FeatureCollection from a geopandas dataframe.
        Associates start and end times to the shape, if present, and
        stores the remaining columns as shape properties.

        Args:
            df:
                A GeoPandas dataframe
            time_start_field:
                The field name for the start time
            time_end_field:
                The field name for the end time. If a start time is present
                but an end time is not, this value will default to the
                start time.

        Returns:
            An object of this class's type
        """
        import geopandas as gpd
        import pandas as pd

        conv_map = {
            'Point': GeoPoint,
            'LineString': GeoLineString,
            'Polygon': GeoPolygon,
            'MultiPoint': MultiGeoPoint,
            'MultiLineString': MultiGeoLineString,
            'MultiPolygon': MultiGeoPolygon,
        }

        def _get_dt(rec):
            """Grabs datetime data and returns appropriate struct"""
            dt_start = rec.get(time_start_field)
            dt_end = rec.get(time_end_field)
            if not (
                (not pd.isnull(dt_start) and isinstance(dt_start, datetime)) or
                (not pd.isnull(dt_end) and isinstance(dt_end, datetime))
            ):
                return None

            if not (dt_start and dt_end) or dt_start == dt_end:
                return dt_start or dt_end

            return TimeInterval(dt_start, dt_end)

        df = cast(gpd.GeoDataFrame, df)
        prop_fields = [
            x for x in df.columns if x not in (time_start_field, time_end_field, 'geometry')
        ]
        shapes: List[BaseShape] = []
        for record in df.to_dict('records'):
            geom_type = record['geometry'].geom_type
            if geom_type not in conv_map:  # pragma: no cover
                # ignored coverage because can't falsify geometry type
                raise ValueError(f'Unrecognized geometry type: {geom_type}')

            dt = _get_dt(record)
            props = {k: v for k, v in record.items() if k in prop_fields}
            shapes.append(
                conv_map[geom_type].from_wkt(  # type: ignore
                    record['geometry'].wkt,
                    dt=dt,
                    properties=props
                )
            )

        return cls(shapes)

    @classmethod
    def from_shapefile(
        cls,
        zip_fpath: Union[str, Path],
        time_start_field: str = 'datetime_s',
        time_end_field: str = 'datetime_e',
        read_layers: Optional[List[str]] = None
    ):
        import shapefile

        def _get_dt(rec):
            """Grabs datetime data and returns appropriate struct"""
            # Convert empty strings to None
            dt_start = rec.get(time_start_field) or None
            dt_end = rec.get(time_end_field) or None
            if dt_start is None and dt_end is None:
                return None

            if dt_start:
                dt_start = datetime.fromisoformat(dt_start)

            if dt_end:
                dt_end = datetime.fromisoformat(dt_end)

            if not (dt_start and dt_end) or dt_start == dt_end:
                return dt_start or dt_end

            return TimeInterval(dt_start, dt_end)

        conv_map = {
            'Point': GeoPoint,
            'LineString': GeoLineString,
            'Polygon': GeoPolygon,
            'MultiPoint': MultiGeoPoint,
            'MultiLineString': MultiGeoLineString,
            'MultiPolygon': MultiGeoPolygon,
        }

        shapes = []
        with ZipFile(zip_fpath, 'r') as z:
            files_in_zip = z.namelist()

        for file_name in files_in_zip:
            if read_layers and file_name.split('.')[0] not in read_layers:
                continue

            if not file_name.endswith('.shp'):
                continue

            reader = shapefile.Reader(Path(zip_fpath) / file_name)
            if not reader.shapes():  # pragma: no cover
                # Layer is empty
                continue

            for shape, record in zip(reader.shapes(), reader.records()):
                arc_type = shape.__geo_interface__.get('type')
                if arc_type not in conv_map:  # pragma: no cover
                    raise ValueError(
                        f'Shapefile contains unsupported shape type: {reader.shapeTypeName}'
                    )

                geostructs_type = conv_map[arc_type]

                props = record.as_dict()
                dt = _get_dt(props)
                props = {
                    k: v for k, v in props.items() if k not in (time_start_field, time_end_field)
                }

                shapes.append(
                    geostructs_type.from_pyshp(shape, dt=dt, properties=props)  # type: ignore
                )

        return cls(shapes)

    @classmethod
    def from_shapely(cls, geometry_collection):
        """
        Creates a geostructures FeatureCollection from a shapely GeometryCollection

        Args:
            geometry_collection:
                A shapely GeometryCollection

        Returns:
            FeatureCollection
        """
        conv_map = {
            'Point': GeoPoint,
            'LineString': GeoLineString,
            'Polygon': GeoPolygon,
            'MultiPoint': MultiGeoPoint,
            'MultiLineString': MultiGeoLineString,
            'MultiPolygon': MultiGeoPolygon,
        }

        shapes = []
        for shape in geometry_collection.geoms:
            geom_type = shape.geom_type
            if geom_type not in conv_map:  # pragma: no cover
                # ignored coverage because can't falsify geometry type
                raise ValueError(f'Unrecognized geometry type: {geom_type}')

            shapes.append(
                conv_map[geom_type].from_shapely(shape)
            )

        return FeatureCollection(shapes)

    @cached_property
    def geospan(self) -> float:
        """
        A summary statistic equal to the width of self.bounds in degrees
        plus the height of self.bounds in degrees. Can be used as a quick
        way to sort larger (in extent) FeatureCollections from smaller ones.
        """
        bounds = self.bounds
        return bounds[2] - bounds[0] + bounds[3] - bounds[1]

    def intersects(self, shape: BaseShape):
        """
        Boolean determination of whether any pings from the track exist inside the provided
        geostructure.

        Args:
            shape:
                A geostructure from geostructure.geostructures

        Returns:
            bool
        """
        shapes = self.geoshapes
        if shape.dt:
            shapes = self.filter_by_dt(shape.dt).geoshapes

        for col_shape in shapes:
            if col_shape.intersects(shape):
                return True

        return False

    def to_fastkml_folder(self, folder_name: str):
        from fastkml import Folder

        return Folder(
            name=folder_name,
            features=[x.to_fastkml_placemark() for x in self.geoshapes]
        )

    def to_geojson(self, properties: Optional[Dict] = None, **kwargs):
        return {
            'type': 'FeatureCollection',
            'features': [
                x.to_geojson(
                    properties=properties,
                    id=idx,  # default to idx, but overridden by kwargs if specified
                    **kwargs
                ) for idx, x in enumerate(self.geoshapes)
            ]
        }

    def to_geopandas(self, include_properties: Optional[List[str]] = None):
        """
        """
        import pandas as pd
        import geopandas as gpd

        keys = include_properties or set(
            _key for x in self.geoshapes
            for _key in x.properties.keys()
        )

        return gpd.GeoDataFrame(
            data=pd.DataFrame(
                [
                    {
                       key: x.properties.get(key) for key in keys
                    } for x in self.geoshapes
                ]
            ),
            geometry=gpd.GeoSeries.from_wkt([x.to_wkt() for x in self.geoshapes])
        )

    def to_shapefile(
        self,
        zip_file: ZipFile,
        include_properties: Optional[List[str]] = None,
    ) -> None:
        """
        Write the collection to a zip-archived ESRI shapefile set.

        Each geometry family (points, multipoints, lines, polygons) is **further
        split** by its dimensionality (XY / XYM / XYZ) so that every writer
        receives only *one* shape type – required by pyshp.
        """
        import shapefile

        # 1. Split the collection into geometry families
        def _iter_coords(geom) -> Iterable["Coordinate"]:
            """Yield every Coordinate contained in *geom* (recursively)."""
            if isinstance(geom, GeoPoint):
                yield geom.centroid

            elif isinstance(geom, MultiGeoPoint):
                for pt in geom.geoshapes:
                    yield pt.centroid

            elif isinstance(geom, GeoLineString):
                yield from geom.vertices

            elif isinstance(geom, MultiGeoLineString):
                for line in geom.geoshapes:
                    yield from line.vertices

            elif isinstance(geom, PolygonLikeMixin):
                for ring in geom.linear_rings():
                    yield from ring

            else:  # pragma: no cover  # fallback – should not happen
                return

        def _dimensionality(geom) -> str:
            """Return 'XYZ' (has Z), 'XYM' (has M, no Z) or 'XY'."""
            has_z = has_m = False
            for c in _iter_coords(geom):
                has_z |= getattr(c, "z", None) is not None
                has_m |= getattr(c, "m", None) is not None
                if has_z:  # once True the result cannot change
                    break
            if has_z:
                return "XYZ"
            if has_m:
                return "XYM"
            return "XY"

        def _convert_dt(val: Any) -> Any:
            """Convert date / datetime to ISO strings so dBASE does not choke."""
            return val.isoformat() if isinstance(val, (date, datetime)) else val

        class Families(TypedDict):
            points: list[GeoPoint]
            multipoints: list[MultiGeoPoint]
            lines: list[LineLikeMixin]
            shapes: list[PolygonLikeMixin]

        families: Families = {
            "points": [],
            "multipoints": [],
            "lines": [],
            "shapes": [],
        }
        for g in self.geoshapes:
            if isinstance(g, GeoPoint):
                families["points"].append(g)
            elif isinstance(g, MultiGeoPoint):
                families["multipoints"].append(g)
            elif isinstance(g, LineLikeMixin):
                families["lines"].append(g)
            elif isinstance(g, PolygonLikeMixin):
                families["shapes"].append(g)
            else:
                raise ValueError(f"Unrecognised geometry type: {type(g)}")

        # 2. Work in a temp directory so we can zip afterwards
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            for fam_name, geoms in families.items():
                # 2a. Further split by dimensionality
                dims = defaultdict(list)  # key: 'XY' / 'XYM' / 'XYZ'
                for g in cast(Sequence, geoms):
                    dims[_dimensionality(g)].append(g)

                for dim_key, sublist in dims.items():
                    # 2b. Create writer for this (family, dimensionality)
                    base = tmpdir_path / f"{fam_name}_{dim_key.lower()}"
                    with shapefile.Writer(base) as w:
                        # Collect properties & declare dBASE fields
                        dtype_pairs = {
                            (k, type(v))
                            for geom in sublist
                            for k, v in geom.properties.items()
                            if include_properties is None or k in include_properties
                        }
                        dtype_map = dict(dtype_pairs)

                        for field, dtype in dtype_map.items():
                            if issubclass(dtype, bool):
                                w.field(field, "L")
                            elif issubclass(dtype, (int, float)) and not issubclass(dtype, datetime):
                                w.field(field, "N")
                            else:
                                w.field(field, "C")
                        w.field("ID", "N")

                        # Warn about inconsistent property types
                        if dtype_pairs:
                            mc = Counter(k for k, _ in dtype_pairs).most_common(1)[0]
                            if mc[1] > 1:
                                LOGGER.warning(
                                    "Conflicting data types found in properties; "
                                    "your shapefile may not be written correctly."
                                )

                        # Write features
                        for idx, geom in enumerate(sublist):
                            props = geom.properties
                            w.record(*[_convert_dt(props.get(k)) for k in dtype_map], idx)
                            geom.to_pyshp(w)

                    # 2c. Add the four components to the zip
                    for ext in (".shp", ".shx", ".dbf"):
                        comp_path = f"{base}{ext}"
                        zip_file.write(comp_path, f"{base.name}{ext}")

                    prj = tmpdir_path / f"{base.name}.prj"
                    prj.write_text(
                        'GEOGCS["WGS 84",'
                        'DATUM["WGS_1984",'
                        'SPHEROID["WGS 84",6378137,298.257223563]],'
                        'PRIMEM["Greenwich",0],'
                        'UNIT["degree",0.0174532925199433]]'
                    )
                    zip_file.write(prj, prj.name)

        # nothing to return
        return


class FeatureCollection(CollectionBase):

    """
    A collection of GeoShapes, in no particular order
    """

    def __add__(self, other):
        if not isinstance(other, FeatureCollection):
            raise ValueError(
                'You can only combine a FeatureCollection with another FeatureCollection'
            )

        return FeatureCollection(self.geoshapes + other.geoshapes)

    def __eq__(self, other):
        """Test equality"""
        if not isinstance(other, FeatureCollection):
            return False

        if not self.geoshapes == other.geoshapes:
            return False

        return True

    def __getitem__(self, item):
        """Slicing by index"""
        return self.geoshapes.__getitem__(item)

    def __iter__(self):
        """Iterate through the track"""
        return self.geoshapes.__iter__()

    def __len__(self):
        """The track length"""
        return self.geoshapes.__len__()

    def __repr__(self):
        """REPL representation"""
        if not self.geoshapes:
            return '<Empty FeatureCollection>'

        return f'<FeatureCollection with {len(self.geoshapes)} shapes>'

    def copy(self):
        """Returns a shallow copy of self"""
        return FeatureCollection(self.geoshapes.copy())


class Track(CollectionBase):

    """
    A sequence of chronologically-ordered (by start time) GeoShapes
    """

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(self, geoshapes: List[BaseShape]):
        if not all(x.dt for x in geoshapes):
            raise ValueError('All track geoshapes must have an associated time value.')

        super().__init__(sorted(geoshapes, key=lambda x: x.start))

    def __add__(self, other):
        if not isinstance(other, Track):
            raise ValueError('You can only combine a Track with another Track')

        return Track(self.geoshapes + other.geoshapes)

    def __eq__(self, other):
        """Test equality"""
        if not isinstance(other, Track):
            return False

        if not self.geoshapes == other.geoshapes:
            return False

        return True

    def __getitem__(self, val: slice):
        """
        Permits track slicing by datetime.

        Args:
            val:
                A slice of datetimes or a datetime

        Examples:
            ```python
            # Returns all points from 1 JAN 2020 00:00 (inclusive) through
            # 2 JAN 2020 00:00 (not inclusive)
            track[datetime(2020, 1, 1):datetime(2020, 1, 2)
            ```

        Returns:
            Track
        """
        _start = default_to_zulu(
            val.start or self.geoshapes[0].start
        )
        _stop = default_to_zulu(
            val.stop or self.geoshapes[-1].end + timedelta(seconds=1)
        )
        return Track(
            [x for x in self.geoshapes if _start <= x.start and x.end < _stop]
        )

    def __repr__(self):
        """REPL representation"""
        if not self.geoshapes:
            return '<Empty Track>'

        return f'<Track with {len(self.geoshapes)} shapes ' \
               f'from {self.geoshapes[0].start.isoformat()} - ' \
               f'{self.geoshapes[-1].end.isoformat()}>'

    @cached_property
    def centroid_distances(self):
        """Provides an array of the distances (in meters) between chronologically-ordered
        pings. The length of the returned array will always be len(self) - 1"""
        if len(self.geoshapes) < 2:
            raise ValueError('Cannot compute distances between fewer than two pings.')

        return np.array([
            haversine_distance_meters(x.centroid, y.centroid)
            for x, y in zip(self.geoshapes, self.geoshapes[1:])
        ])

    @property
    def end(self):
        """The timestamp of the final ping"""
        if not self.geoshapes:
            raise ValueError('Cannot compute finish time of an empty track.')

        return self.geoshapes[-1].end

    @property
    def first(self):
        """The first ping"""
        if not self.geoshapes:
            raise ValueError('Track has no pings.')

        return self.geoshapes[0]

    @cached_property
    def has_duplicate_timestamps(self):
        """Determine if there are different pings with the same timestamp in the data"""
        _ts = set()
        for point in self.geoshapes:
            if point.dt in _ts:
                return True
            _ts.add(point.dt)
        return False

    @property
    def last(self):
        """The last ping"""
        if not self.geoshapes:
            raise ValueError('Track has no pings.')

        return self.geoshapes[-1]

    @property
    def speed_diffs(self):
        """
        Provides speed differences (meters per second) between pings in a track

        Returns:
            np.Array
        """
        return self.centroid_distances / [x.total_seconds() for x in self.time_start_diffs]

    @property
    def start(self):
        """The timestamp of the first ping"""
        if not self.geoshapes:
            raise ValueError('Cannot compute start time of an empty track.')

        return self.geoshapes[0].start

    @cached_property
    def time_start_diffs(self):
        """Provides an array of the time differences between chronologically-ordered
        pings. The length of the returned array will always be len(self) - 1"""
        if len(self.geoshapes) < 2:
            raise ValueError('Cannot compute time diffs between fewer than two shapes.')

        return np.array([
            (y.start - x.start)
            for x, y in zip(self.geoshapes, self.geoshapes[1:])
        ])

    def copy(self):
        """Returns a shallow copy of self"""
        return Track(self.geoshapes.copy())

    def convolve_duplicate_timestamps(self):
        """Convolves pings with duplicate timestamps and returns a new track"""
        if not self.has_duplicate_timestamps:
            return self.copy()

        # group by timestamp (in future, add a grouping window mechanism)
        _timestamp_grouping = defaultdict(list)
        for point in self.geoshapes:
            _timestamp_grouping[point.dt].append(point)

        # Currently only points are supported, so just average the lon/lats
        new_pings = []
        for _ts, ping_group in _timestamp_grouping.items():
            if len(ping_group) == 1:
                new_pings.append(ping_group[0])
                continue

            _lons, _lats = list(zip(*[x.centroid.to_float() for x in ping_group]))
            new_pings.append(
                GeoPoint(
                    Coordinate(sum(_lons)/len(_lons), sum(_lats)/len(_lats)),
                    _ts,
                    properties={
                        k: v for shape in ping_group
                        for k, v in shape._properties.items()
                    }
                )
            )

        return Track(new_pings)

    def filter_by_time(self, start_time: time, end_time: time) -> 'Track':
        """Filters the track by time of day"""
        return Track(
            [
                shape for shape in self.geoshapes
                if start_time <= shape.end.time() <= end_time
                or start_time <= shape.start.time() <= end_time
                or shape.start.time() <= start_time <= end_time <= shape.end.time()
            ]
        )

    def filter_impossible_journeys(self, max_speed: float) -> 'Track':
        """
        Filters out impossible journeys in the track based on speed thresholds.

        Args:
            max_speed (float): The maximum allowable speed (meters per second).

        Returns:
            Track: A new Track instance with only valid geoshapes, removing impossible journeys.
        """
        # Track shapes are guaranteed to have .start
        times = [shape.start for shape in self.geoshapes]
        coords = [pt.centroid for pt in self.geoshapes]
        i = 0
        valid_geoshapes = [self.geoshapes[i]]  # Keep the first point as valid

        for j in range(1, len(self.geoshapes)):
            # Use pre-baked Haversine calculation
            dx = haversine_distance_meters(coords[i], coords[j])
            dt = (times[j] - times[i]).total_seconds()  # Time difference in seconds
            if dt == 0:
                warn_once(
                    'Duplicate timestamps detected; filtering all but the first. '
                    'This warning will not repeat.'
                )
                continue

            speed = 0 if dx == 0 else dx / dt

            if np.isnan(speed):  # Handle NaN speeds
                i = j  # Move starting point to the current point
            elif speed <= max_speed:
                valid_geoshapes.append(self.geoshapes[j])  # Add valid point to the list
                i = j  # Move starting point to current point

        # Create a new Track with only valid geoshapes
        return Track(valid_geoshapes)

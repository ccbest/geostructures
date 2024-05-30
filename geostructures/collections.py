"""
Module for sequences of GeoShapes
"""

__all__ = ['FeatureCollection', 'ShapeCollection', 'Track']

from collections import defaultdict, Counter
from datetime import date, datetime, time, timedelta
from functools import cached_property
import os
from pathlib import Path
import tempfile
from typing import cast, Any, List, Dict, Optional, Union, Tuple, TypeVar
from zipfile import ZipFile

import numpy as np

from geostructures import Coordinate, LOGGER
from geostructures.structures import GeoLineString, GeoPoint, GeoPolygon
from geostructures._base import BaseShape, LineLike, PointLike, ShapeLike
from geostructures.time import TimeInterval
from geostructures.calc import haversine_distance_meters
from geostructures.utils.functions import default_to_zulu


_COL_TYPE = TypeVar('_COL_TYPE', bound='ShapeCollection')


class ShapeCollection:

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
    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        all_bounds = [x.bounds for x in self.geoshapes]
        return (
            (
                min(x[0][0] for x in all_bounds),
                max(x[0][1] for x in all_bounds),
            ),
            (
                min(x[1][0] for x in all_bounds),
                max(x[1][1] for x in all_bounds)
            )
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
        from scipy import spatial  # pylint: disable=import-outside-toplevel

        if len(self.geoshapes) <= 2 and all(isinstance(x, GeoPoint) for x in self.geoshapes):
            raise ValueError('Cannot create a convex hull from less than three points.')

        _points = filter(lambda x: isinstance(x, GeoPoint), self.geoshapes)
        _lines = cast(
            List[GeoLineString],
            filter(lambda x: isinstance(x, GeoLineString), self.geoshapes)
        )
        _shapes = filter(lambda x: not isinstance(x, (GeoPoint, GeoLineString)), self.geoshapes)

        points = []
        points += [y.to_float() for x in _shapes for y in x.bounding_coords()]
        points += [y.to_float() for x in _lines for y in x.vertices]
        points += [x.centroid.to_float() for x in _points]
        hull = spatial.ConvexHull(points)
        return GeoPolygon([Coordinate(*points[x]) for x in [*hull.vertices, hull.vertices[0]]])

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

        if gjson.get('type') != 'FeatureCollection':
            raise ValueError('Malformed GeoJSON; expected FeatureCollection')

        shapes: List[BaseShape] = []
        for feature in gjson.get('features', []):
            geom_type = feature.get('geometry', {}).get('type')
            if geom_type == 'Point':
                shapes.append(
                    GeoPoint.from_geojson(
                        feature,
                        time_start_property,
                        time_end_property,
                        time_format=time_format
                    )
                )
                continue

            if geom_type == 'LineString':
                shapes.append(
                    GeoLineString.from_geojson(
                        feature,
                        time_start_property,
                        time_end_property,
                        time_format=time_format
                    )
                )
                continue

            if geom_type == 'Polygon':
                shapes.append(
                    GeoPolygon.from_geojson(
                        feature,
                        time_start_property,
                        time_end_property,
                        time_format=time_format
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
            dt = _get_dt(record)
            props = {k: v for k, v in record.items() if k in prop_fields}
            if record['geometry'].geom_type == 'Point':
                shapes.append(GeoPoint.from_wkt(record['geometry'].wkt, dt, props))
                continue

            if record['geometry'].geom_type == 'LineString':
                shapes.append(
                    GeoLineString.from_wkt(record['geometry'].wkt, dt, props)
                )
                continue

            if record['geometry'].geom_type == 'Polygon':
                shapes.append(
                    GeoPolygon.from_wkt(record['geometry'].wkt, dt, props)
                )
                continue

        return cls(shapes)

    @classmethod
    def from_shapefile(
        cls,
        zip_fpath: Union[str, Path],
        time_start_field: str = 'datetime_s',
        time_end_field: str = 'datetime_e',
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

        def _create_point(shape, dt, props):
            return GeoPoint(Coordinate(*shape.points[0]), dt=dt, properties=props)

        def _create_polygon(shape, dt, props):
            """
            Create a polygon out of a pyshyp polygon. Note that "points" are continuous across
            the bounding vertices and holes, so if multiple "parts" are present we need to segment
            the list of points. "parts" will only provide the indices for segmentation.
            """
            parts = list(shape.parts)
            if parts == [0]:
                return GeoPolygon([Coordinate(*x) for x in shape.points], dt=dt, properties=props)

            rings = [
                [Coordinate(*x) for x in shape.points[start: stop if stop > 0 else None]]
                for start, stop in zip(shape.parts, [*shape.parts[1:], -1])
            ]
            holes = [GeoPolygon(x[::-1]) for x in rings[1:]]
            return GeoPolygon(rings[0], holes=holes, dt=dt, properties=props)

        def _create_linestring(shape, dt, props):
            return GeoLineString([Coordinate(*x) for x in shape.points], dt=dt, properties=props)

        shapes = []
        with ZipFile(zip_fpath, 'r') as z:
            files_in_zip = z.namelist()

        for file_name in files_in_zip:
            if not file_name.endswith('.shp'):
                continue

            reader = shapefile.Reader(Path(zip_fpath) / file_name)

            type_map = {
                'POLYLINE': _create_linestring,
                'POINT': _create_point,
                'POLYGON': _create_polygon,
            }
            shape_fn = type_map.get(reader.shapeTypeName)
            if not shape_fn:  # pragma: no cover
                raise ValueError(
                    f'Shapefile contains unsupported shape type: {reader.shapeTypeName}'
                )

            for shape, record in zip(reader.shapes(), reader.records()):
                props = record.as_dict()
                dt = _get_dt(props)
                props = {k: v for k, v in props.items() if k not in (time_start_field, time_end_field)}
                shapes.append(shape_fn(shape, dt=dt, props=props))

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
        from shapely.geometry import LineString, Point, Polygon

        shapes = []
        for shape in geometry_collection.geoms:
            if isinstance(shape, Point):
                shapes.append(GeoPoint.from_shapely(shape))
                continue

            if isinstance(shape, Polygon):
                shapes.append(GeoPolygon.from_shapely(shape))
                continue

            if isinstance(shape, LineString):
                shapes.append(GeoLineString.from_shapely(shape))

        return FeatureCollection(shapes)

    @cached_property
    def geospan(self) -> float:
        """
        A summary statistic equal to the width of self.bounds in degrees
        plus the height of self.bounds in degrees. Can be used as a quick
        way to sort larger (in extent) FeatureCollections from smaller ones.
        """
        bounds = self.bounds
        return bounds[0][1] - bounds[0][0] + bounds[1][1] - bounds[1][0]

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
        Writes the collection to a ESRI shapefile. Note that shape"file"s actually
        consist of several files, so you should provide a directory rather than
        a file name.

        Requires the pyshp library (canonically imported as 'shapefile').

        Args:
            zip_file:
                The zipfile to write shape files to.

            include_properties: (Default None)
                A list of properties to include. If None, all properties will be included.

        Returns:
            None
        """
        import shapefile

        def _convert_dt(val: Any):
            """Convert date/datetime values to string"""
            if isinstance(val, (datetime, date)):
                return val.isoformat()
            return val

        points: List[BaseShape] = []
        lines: List[BaseShape] = []
        shapes: List[BaseShape] = []
        for shape in self.geoshapes:
            if isinstance(shape, GeoPoint):
                points.append(shape)

            elif isinstance(shape, GeoLineString):
                lines.append(shape)

            else:
                shapes.append(shape)

        with tempfile.TemporaryDirectory() as tempdir:
            for shapetype, shape_group in (('points', points), ('lines', lines), ('shapes', shapes)):
                if not shape_group:
                    continue

                writer = shapefile.Writer(os.path.join(tempdir, shapetype))

                # 2-Tuples of properties and their datatypes
                _types = set(
                    (_key, type(_val)) for x in shape_group
                    for _key, _val in x.properties.items()
                    if (not include_properties) or _key in include_properties
                )
                typemap = dict(_types)

                # Declare fields
                for field, _type in typemap.items():
                    if issubclass(_type, bool):
                        # bools are subclasses of int (WAT) - have to check first
                        writer.field(field, 'L')
                    elif issubclass(_type, (float, int)) and not issubclass(_type, datetime):
                        # datetimes are subclasses of ints too
                        writer.field(field, 'N')
                    else:
                        writer.field(field, 'C')

                writer.field('ID', 'N')

                if _types:
                    # Check to make sure data types are consistent
                    c = Counter(x[0] for x in _types)
                    if c.most_common(1)[0][1] > 1:
                        # Just log a warning - still want to try to write the file
                        LOGGER.warning(
                            'Conflicting data types found in properties; '
                            'your shapefile may not get written correctly'
                        )

                # Write shapes to file
                for idx, shape in enumerate(shape_group):
                    # Write out properties
                    props = shape.properties
                    writer.record(*[_convert_dt(props.get(k)) for k in typemap], idx)

                    if isinstance(shape, PointLike):
                        writer.point(*shape.centroid.to_float())

                    elif isinstance(shape, LineLike):
                        writer.line([[list(x.to_float()) for x in shape.vertices]])

                    else:
                        writer.poly(
                            [
                                [list(coord.to_float()) for coord in ring]
                                for ring in shape.linear_rings()
                            ]
                        )

                writer.close()
                zip_file.write(writer.shx.name, writer.shx.name.split(os.sep)[-1])
                zip_file.write(writer.shp.name, writer.shp.name.split(os.sep)[-1])
                zip_file.write(writer.dbf.name, writer.dbf.name.split(os.sep)[-1])

                with open(os.path.join(tempdir, f'{shapetype}.prj'), 'w+') as f:
                    # Taken from pyshp's readme
                    wkt = 'GEOGCS["WGS 84",'
                    wkt += 'DATUM["WGS_1984",'
                    wkt += 'SPHEROID["WGS 84",6378137,298.257223563]]'
                    wkt += ',PRIMEM["Greenwich",0],'
                    wkt += 'UNIT["degree",0.0174532925199433]]'
                    f.write(wkt)
                    zip_file.write(f.name, f.name.split(os.sep)[-1])

        return


class FeatureCollection(ShapeCollection):

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


class Track(ShapeCollection):

    """
    A sequence of chronologically-ordered (by start time) GeoShapes
    """

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
                    _ts
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

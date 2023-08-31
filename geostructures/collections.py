"""
Module for sequences of GeoShapes
"""

__all__ = ['FeatureCollection', 'Track']

from collections import defaultdict
from datetime import date, datetime, timedelta
from functools import cached_property
from typing import Any, List, Dict, Optional, Union

import numpy as np

from geostructures.coordinates import Coordinate
from geostructures.structures import GeoShape, GeoPoint, GeoPolygon
from geostructures.time import TimeInterval
from geostructures.calc import haversine_distance_meters
from geostructures.utils.mixins import LoggingMixin, DefaultZuluMixin


class ShapeCollection(LoggingMixin, DefaultZuluMixin):

    geoshapes: List[GeoShape]

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

    @cached_property
    def convex_hull(self):
        """Creates a convex hull around the pings"""
        from scipy import spatial

        if len(self.geoshapes) <= 2 and all(isinstance(x, GeoPoint) for x in self.geoshapes):
            raise ValueError('Cannot create a convex hull from less than three points.')

        points = [y.to_float() for x in self.geoshapes for y in x.bounding_coords()]
        hull = spatial.ConvexHull(points)
        return GeoPolygon([Coordinate(*points[x]) for x in hull.vertices])


class FeatureCollection(ShapeCollection):

    """
    A collection of GeoShapes, in no particular order
    """

    def __init__(self, geoshapes: List[GeoShape]):
        super().__init__()
        self.geoshapes = geoshapes

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


class Track(ShapeCollection, LoggingMixin, DefaultZuluMixin):

    """
    A sequence of chronologically-ordered (by start time) GeoShapes
    """

    def __init__(self, geoshapes: List[GeoShape], metadata: Optional[Dict[str, Any]] = None):
        super().__init__()
        if not all(x.dt for x in geoshapes):
            raise ValueError('All track geoshapes must have an associated time value.')

        self.geoshapes = sorted(geoshapes, key=lambda x: x.start)
        self.metadata = metadata or {}

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

    def __getitem__(self, val: Union[slice, datetime]):
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
        if isinstance(val, datetime):
            val = self._default_to_zulu(val)
            return Track(
                [x for x in self.geoshapes if x.dt == val]
            )

        _start = self._default_to_zulu(
            val.start or self._date_to_datetime(self.geoshapes[0].start)
        )
        _stop = self._default_to_zulu(
            val.stop or self._date_to_datetime(self.geoshapes[-1].end + timedelta(seconds=1))
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

    def copy(self):
        """Returns a shallow copy of self"""
        return Track(self.geoshapes.copy())

    @property
    def first(self):
        """The first ping"""
        if not self.geoshapes:
            raise ValueError('Track has no pings.')

        return self.geoshapes[0]

    @property
    def last(self):
        """The last ping"""
        if not self.geoshapes:
            raise ValueError('Track has no pings.')

        return self.geoshapes[-1]

    @property
    def start(self):
        """The timestamp of the first ping"""
        if not self.geoshapes:
            raise ValueError('Cannot compute start time of an empty track.')

        return self.geoshapes[0].dt

    @property
    def finish(self):
        """The timestamp of the final ping"""
        if not self.geoshapes:
            raise ValueError('Cannot compute finish time of an empty track.')

        return self.geoshapes[-1].dt

    @property
    def speed_diffs(self):
        """
        Provides speed differences (meters per second) between pings in a track

        Returns:
            np.Array
        """
        return self.centroid_distances / [x.total_seconds() for x in self.time_start_diffs]

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
    def time_start_diffs(self):
        """Provides an array of the time differences between chronologically-ordered
        pings. The length of the returned array will always be len(self) - 1"""
        if len(self.geoshapes) < 2:
            raise ValueError('Cannot compute time diffs between fewer than two shapes.')

        return np.array([
            (y.start - x.start)
            for x, y in zip(self.geoshapes, self.geoshapes[1:])
        ])

    @cached_property
    def has_duplicate_timestamps(self):
        """Determine if there are different pings with the same timestamp in the data"""
        _ts = set()
        for point in self.geoshapes:
            if point.dt in _ts:
                return True
            _ts.add(point.dt)
        return False

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

            _lons, _lats = list(zip(*[x.center.to_float() for x in ping_group]))
            new_pings.append(
                GeoPoint(
                    Coordinate(sum(_lons)/len(_lons), sum(_lats)/len(_lats)),
                    _ts
                )
            )

        return Track(new_pings, self.metadata)

    @staticmethod
    def _date_to_datetime(dt: Union[date, datetime]) -> datetime:
        """Converts a date into datetime (assumes midnight)"""
        if isinstance(dt, datetime):
            return dt
        return datetime(dt.year, dt.month, dt.day)

    def _subset_by_dt(self, dt: Union[date, datetime, TimeInterval]):
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
            return self[dt]  # type: ignore

        if isinstance(dt, date):
            _start = self._date_to_datetime(dt)
            _end = self._date_to_datetime(dt + timedelta(days=1))
            return self[_start:_end]  # type: ignore

        if isinstance(dt, TimeInterval):
            _start = dt.start
            _end = dt.end
            return self[_start:_end]  # type: ignore

        raise ValueError(f"Unexpected dt object: {dt}")

    def intersects(self, shape: GeoShape):
        """
        Boolean determination of whether any pings from the track exist inside the provided
        geostructure.

        Args:
            shape:
                A geostructure from geostructure.geostructures

        Returns:
            bool
        """
        points = self.geoshapes
        if shape.dt:
            points = self._subset_by_dt(shape.dt).geoshapes

        for point in points:
            if point.centroid in shape:
                return True

        return False

    def intersection(self, shape: GeoShape):
        """
        Returns the subset of the track which exists inside the provided geostructure.

        Args:
            shape:
                A geostructure from geostructure.geostructures

        Returns:
            Track
        """
        points = self.geoshapes
        if shape.dt:
            points = self._subset_by_dt(shape.dt).geoshapes

        return Track([point for point in points if point.centroid in shape])

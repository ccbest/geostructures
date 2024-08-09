"""
Representation of a specific point on earth
"""

__all__ = ['Coordinate']

from functools import cached_property
import math
from typing import List, Tuple, Union

from geostructures.utils.functions import round_half_up


class Coordinate:
    """Representation of a coordinate on the globe (i.e., a lon/lat pair)"""

    def __init__(
        self,
        longitude: Union[float, int, str],
        latitude: Union[float, int, str],
        _bounded: bool = True
    ):
        lon, lat = float(longitude), float(latitude)
        if _bounded:
            while not -90 <= lat <= 90:
                # Crosses one of the poles
                lat = 90 - (lat - 90) if lat > 90 else -90 - (lat + 90)
                lon = lon + 180 if lon < 0 else lon - 180

            while not -180 <= lon <= 180:
                # Crosses the antimeridian
                lon = lon - 360 if lon > 180 else lon + 360

        # Longitudes are bounded to [-180, 180)
        if lon == 180:
            lon = -180

        self.longitude = lon
        self.latitude = lat

    def __eq__(self, other):
        if not isinstance(other, Coordinate):
            return False

        return self.latitude == other.latitude and self.longitude == other.longitude

    def __hash__(self):
        return hash((self.longitude, self.latitude))

    def __repr__(self):
        return f'<Coordinate({self.longitude}, {self.latitude})>'

    @cached_property
    def xyz(self):
        """Converts lat/lon to unit coordinates [x,y,z]"""
        r_lat = math.radians(self.latitude)
        r_lon = math.radians(self.longitude)
        return [
            math.cos(r_lat) * math.cos(r_lon),
            math.cos(r_lat) * math.sin(r_lon),
            math.sin(r_lat)
        ]

    @classmethod
    def _from_xyz(cls, xyz: List[float]):
        assert len(xyz) == 3
        latitude = math.asin(xyz[2])
        longitude = math.atan2(xyz[1], xyz[0])
        return Coordinate(math.degrees(longitude), math.degrees(latitude))

    @classmethod
    def from_dms(cls, lon: Tuple[int, int, float, str], lat: Tuple[int, int, float, str]):
        """
        Creates a Coordinate from a Degree Minutes Seconds (lon, lat) pair.

        The quadrant value should consist of either 'E'/'W' (longitude) or 'N'/'S' (latitude)

        Args:
            lon:
                Longitude, as a 4-tuple of
                ( <degrees> (float),  <minutes> (float), <seconds> (float), <quadrant> (str))
            lat:
                Latitude, as a 4-tuple of
                ( <degrees> (float),  <minutes> (float), <seconds> (float), <quadrant> (str) )

        Returns:
            Coordinate
        """
        def convert(dms: Tuple[int, int, float, str]):
            mult = -1 if dms[3] in ('S', 'W') else 1
            return mult * (dms[0] + (dms[1] / 60) + (dms[2] / 3600))

        return Coordinate(convert(lon), convert(lat))

    @classmethod
    def from_mgrs(cls, mgrs_str: str):
        """Create a Coordinate object from a MGRS string"""
        import mgrs  # pylint: disable=import-outside-toplevel
        _MGRS = mgrs.MGRS()

        # Spaces in the mgrs string can produce inaccurate coordinates
        lat, lon = _MGRS.toLatLon(mgrs_str.replace(' ', ''))
        return Coordinate(lon, lat)

    @classmethod
    def from_projection(cls, lon: float, lat: float, crs: str):
        """
        Creates a Coordinate from a lon, lat pair
        in a different projection than WGS84
        Args:
            lon:
                The longitude

            lat:
                The latitude
            crs:
                A string representing the target EPSG code. e.g EPSG:3857

        """
        from pyproj import Transformer
        transformer = Transformer.from_crs(crs, 'EPSG:4326')
        x, y = transformer.transform(lat, lon)

        return Coordinate(
            round_half_up(y, 6),
            round_half_up(x, 6),
        )

    @classmethod
    def from_qdms(cls, lon: str, lat: str):
        """
        Creates a Coordinate from a QDDMMSSHH (lon, lat) pair

        Args:
            lon:
                The longitude, ex. 'N001140442'

            lat:
                The latitude, ex. 'E01140442'
        """
        def convert(q: str, d: str, m: str, s: str):
            return (float(d) + float(m) / 60 + float(s[:2] + '.' + s[2:]) / 3600) * (
                -1 if q in ('W', 'S') else 1
            )
        lon_dms = lon[1:4], lon[4:6], lon[6:]
        lat_dms = lat[1:3], lat[3:5], lat[5:]

        return Coordinate(
            round_half_up(convert(lon[0], *lon_dms), 6),
            round_half_up(convert(lat[0], *lat_dms), 6)
        )

    def to_dms(self) -> Tuple[Tuple[int, int, float, str], Tuple[int, int, float, str]]:
        """
        Convert a value (latitude or longitude) in decimal degrees to a tuple of
        degrees, minutes, seconds, hemisphere

        Returns:
            converted value as (degrees, minutes, seconds, hemisphere)
        """
        def convert(dd: float) -> Tuple[int, int, float]:
            """Converts a Decimal Degree to Degrees Minutes Seconds"""
            minutes, seconds = divmod(abs(dd) * 3600, 60)
            degrees, minutes = divmod(minutes, 60)
            return int(degrees), int(minutes), round_half_up(seconds, 5)

        return (
            (*convert(self.longitude), 'E' if self.longitude >= 0 else 'W'),
            (*convert(self.latitude), 'N' if self.latitude >= 0 else 'S'),
        )

    def to_float(self) -> Tuple[float, float]:
        """Converts the coordinate to a 2-tuple of floats (longitude, latitude)"""
        return self.longitude, self.latitude

    def to_mgrs(self) -> str:
        """Convert this coordinate to a MGRS string"""
        import mgrs  # pylint: disable=import-outside-toplevel
        _MGRS = mgrs.MGRS()

        return _MGRS.toMGRS(self.latitude, self.longitude)

    def to_projection(self, crs: str):
        """
        Reproject a coordinate from the WGS84 projection to another.

        Args:
           crs:
           A string representing the target EPSG code e,g EPSG:3857.

        Return:
            A coordinate in the target projection system.
        """
        from pyproj import Transformer
        transformer = Transformer.from_crs('EPSG:4326', crs)
        x, y = transformer.transform(self.latitude, self.longitude)

        return Coordinate(
            round_half_up(y, 6),
            round_half_up(x, 6),
            False
        )

    def to_qdms(self) -> Tuple[str, str]:
        """
        Converts this coordinate to QDDMMSSHH format

        Returns:
            Tuple[str, str] representing longitude, latitude
        """
        def zero_pad(num: Union[float, int], length: int) -> str:
            """Stringifies a number, removes decimal, and pads zeros to the prefix"""
            _ = str(num).replace('.', '')
            return '0'*(length-len(_))+_

        lon, lat = self.to_dms()
        _lon = [
            zero_pad(abs(lon[0]), 3),
            zero_pad(lon[1], 2),
            zero_pad(round_half_up(lon[2], 2), 4),
        ]
        _lat = [
            zero_pad(abs(lat[0]), 2),
            zero_pad(lat[1], 2),
            zero_pad(round_half_up(lat[2], 2), 4)
        ]

        return f'{lon[3]}{"".join(_lon)}', f'{lat[3]}{"".join(_lat)}'

    def to_str(self) -> Tuple[str, str]:
        """Converts the coordinate to a 2-tuple of strings (longitude, latitude)"""
        return str(self.longitude), str(self.latitude)

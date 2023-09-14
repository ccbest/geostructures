"""
Representation of a specific point on earth
"""

__all__ = ['Coordinate']

from dataclasses import dataclass
from typing import Optional, Tuple, Union

from precisenumbers import PreciseNumber, parse_number

from geostructures.utils.functions import round_half_up


class Longitude(PreciseNumber):
    """PreciseNumber to represent a longitude value"""

    MAXIMUM = 180
    MINIMUM = -180


class Latitude(PreciseNumber):
    """PreciseNumber to represent a latitude value"""

    MAXIMUM = 90
    MINIMUM = -90


@dataclass
class Coordinate:
    """Representation of a coordinate on the globe (i.e., a lon/lat pair) using PreciseNumbers"""

    longitude: Longitude
    latitude: Latitude

    def __init__(
        self,
        longitude: Union[float, int, str],
        latitude: Union[float, int, str],
        precision: Optional[int] = None,
        same_precision: bool = True,
    ):
        if precision and not same_precision:
            raise ValueError('`precision` cannot be set if `same_precision=False`')

        if same_precision and not precision:
            _, _, _, inferred_lon_precision = parse_number(longitude)
            _, _, _, inferred_lat_precision = parse_number(latitude)
            precision = max(inferred_lon_precision, inferred_lat_precision)

        self.longitude = Longitude(number=longitude, precision=precision)
        self.latitude = Latitude(number=latitude, precision=precision)

    def __hash__(self):
        return hash(
            (
                str(self.longitude),
                self.longitude.precision,
                str(self.latitude),
                self.latitude.precision,
            )
        )

    def __repr__(self):
        return f'<Coordinate({self.longitude}, {self.latitude})>'

    @property
    def precision(self):
        """Returns the minimum precision of the coordinate's 2 precise numbers"""
        return min(self.latitude.precision, self.longitude.precision)

    def to_float(self) -> Tuple[float, float]:
        """Converts the coordinate to a 2-tuple of floats (longitude, latitude)"""
        return float(self.longitude), float(self.latitude)

    def to_str(self) -> Tuple[str, str]:
        """Converts the coordinate to a 2-tuple of strings (longitude, latitude)"""
        return str(self.longitude), str(self.latitude)

    def to_mgrs(self) -> str:
        """Convert this coordinate to a MGRS string"""
        import mgrs  # pylint: disable=import-outside-toplevel
        _MGRS = mgrs.MGRS()

        return _MGRS.toMGRS(float(self.latitude), float(self.longitude))

    @classmethod
    def from_mgrs(cls, mgrs_str: str):
        """Create a Coordinate object from a MGRS string"""
        import mgrs  # pylint: disable=import-outside-toplevel
        _MGRS = mgrs.MGRS()

        lat, lon = _MGRS.toLatLon(mgrs_str)
        return Coordinate(lon, lat)

    def to_dms(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Convert a value (latitude or longitude) in decimal degrees to a tuple of
        degrees, minutes, seconds

        Returns:
            converted value as (degrees, minutes, seconds)
        """
        def convert(dd: float):
            """Converts a Decimal Degree to Degrees Minutes Seconds"""
            is_positive = dd >= 0
            dd = abs(dd)

            minutes, seconds = divmod(dd * 3600, 60)
            degrees, minutes = divmod(minutes, 60)
            degrees = degrees if is_positive else -degrees

            return int(degrees), int(minutes), round_half_up(seconds, 5)

        return convert(float(self.longitude)), convert(float(self.latitude))

    @classmethod
    def from_dms(cls, lon: Tuple[float, float, float], lat: Tuple[float, float, float]):
        """
        Creates a Coordinate from a Degree Minutes Seconds (lon, lat) pair

        Args:
            lon:
                Longitude, as a 3-tuple of
                ( <degrees> (float),  <minutes> (float), <seconds> (float) )
            lat:
                Latitude, as a 3-tuple of
                ( <degrees> (float),  <minutes> (float), <seconds> (float) )

        Returns:
            Coordinate
        """
        def convert(dms: Tuple[float, float, float]):
            return dms[0] + (dms[1] / 60) + (dms[2] / 3600)

        return Coordinate(convert(lon), convert(lat))

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
        lon_q, lat_q = 'S' if lon[0] < 0 else 'N', 'W' if lat[0] < 0 else 'E'
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

        return f'{lon_q}{"".join(_lon)}', f'{lat_q}{"".join(_lat)}'

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

        return Coordinate(convert(lon[0], *lon_dms), convert(lat[0], *lat_dms))

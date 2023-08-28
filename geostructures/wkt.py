
import re

from geostructures import Coordinate



_RE_COORD_GROUP_STR = r'\(((?:\s?\d+\.?\d*\s\d+\.?\d*\,?)+)\)'
_RE_COORD_GROUP = re.compile(_RE_COORD_GROUP_STR)
_RE_POLYGON_WKT = re.compile(r'POLYGON\s?\(' + _RE_COORD_GROUP_STR + r'\)')


def _parse_wkt_coord_group(group: str):
    return [
        Coordinate(*coord.strip().split(' '))
        for coord in group.split(',') if coord
    ]




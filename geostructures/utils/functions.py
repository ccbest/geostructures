"""Module for miscellaneous multi-use functions"""

__all__ = [
    'default_to_zulu', 'get_dt_from_geojson_props', 'is_sub_list',
    'round_half_up', 'sanitize_json'
]

from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, localcontext
from typing import Dict, Any, Optional, Union, List

from geostructures.utils.logging import warn_once
from geostructures.time import TimeInterval


def default_to_zulu(dt: datetime) -> datetime:
    """Add Zulu/UTC as timezone, if timezone not present"""
    if not dt.tzinfo:
        warn_once(
            'Datetime does not contain timezone information; Zulu/UTC time assumed. '
            '(this warning will not repeat)'
        )
        return dt.replace(tzinfo=timezone.utc)

    return dt


def get_dt_from_geojson_props(
    rec: Dict[str, Any],
    time_start_field: str = 'datetime_start',
    time_end_field: str = 'datetime_end',
    time_format: Optional[str] = None
) -> Union[datetime, TimeInterval, None]:
    """Grabs datetime data and returns appropriate struct"""
    def _convert(dt: Optional[str], _format: Optional[str] = None):
        if not dt:
            return None

        if _format:
            return datetime.strptime(dt, _format)

        if dt.endswith('Z'):
            # datetime.fromisoformat can't parse the Z suffix until py3.11
            dt = dt[:-1] + '+00:00'

        return datetime.fromisoformat(dt)

    # Pop the field so it doesn't remain in properties
    dt_start = _convert(rec.pop(time_start_field, None), time_format)
    dt_end = _convert(rec.pop(time_end_field, None), time_format)

    if dt_start is None and dt_end is None:
        return None

    if not (dt_start and dt_end):
        return dt_start or dt_end

    return TimeInterval(dt_start, dt_end)


def round_half_up(value: float, precision) -> float:
    """
    Rounds a value to a given number of decimal places. A value exactly half way
    between two candidates is rounded away from zero, so 0.5 rounds to 1.0 and -0.5
    rounds to -1.0. This differs from the builtin round(), which rounds halves to the
    nearest even candidate.

    Rounding is applied to the decimal value the float displays as, not to its exact
    binary expansion, so round_half_up(2.675, 2) is 2.68 rather than the 2.67 the
    stored value 2.67499999999999982... would otherwise give.

    Args:
        value:
            The float value to be rounded
        precision:
            The number of decimal places to round the float value to

    Returns:
        (float) the rounded value
    """
    with localcontext() as context:
        # Wide enough that quantize never overflows the context for any value a float
        # can hold, at any precision this library rounds to
        context.prec = 60
        return float(
            Decimal(repr(value)).quantize(
                Decimal(1).scaleb(-precision),
                rounding=ROUND_HALF_UP,
            )
        )


def sanitize_json(obj: Any):
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json(x) for x in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


def is_sub_list(list_a: List, list_b: List) -> bool:
    """
    Test whether A is a sublist of B.

    Args:
        list_a: (List)
            A list with elements of Any type

        list_b: (List)
            A second list with elements of Any type

    Returns:
        bool
    """
    if len(list_a) > len(list_b):
        return False

    for i in range(0, len(list_b) - len(list_a) + 1):
        if list_b[i:i+len(list_a)] == list_a:
            return True

    return False

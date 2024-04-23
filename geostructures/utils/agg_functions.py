"""Aggregation functions for use with hash_collection"""

__all__ = ['total_time', 'unique_entities']

from typing import List
from geostructures._base import BaseShape


def total_time(shapes: List[BaseShape]) -> float:
    """
    The total number of seconds elapsed over a list of shapes. Shapes with no time
    bound are not counted.

    Args:
        shapes:
            A list of GeoShapes

    Return:
        float
    """
    return sum(
        [shape.dt.elapsed.total_seconds() if shape.dt else 0 for shape in shapes]
    )


def unique_entities(shapes: List[BaseShape]) -> float:
    """
    The number of distinct entities present in a list of shapes, as represented by the
    'entity' property. Shapes without an entity property will not be counted.

    Args:
        shapes:
            A list of GeoShapes

    Return:
        float
    """
    return float(len(
        set(shape.properties['entity'] for shape in shapes if 'entity' in shape.properties)
    ))

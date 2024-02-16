"""Aggregation functions for use with hash_collection"""

__all__ = ['total_time', 'unique_entities']

from typing import List
from geostructures.structures import GeoShape

def total_time(shapes: List[GeoShape]) -> float:
    """If the hash is larger than the data, this is time spend in each hash in seconds.
    If the hash is smaller than the data, this is time that could have been spent in 
    each hash in seconds."""
    return(sum([(shape.dt.end - shape.dt.start).total_seconds() 
                if shape.dt else 0 for shape in shapes]))

def unique_entities(shapes: List[GeoShape]) -> float:
    """If each shape is attributed to an entity in the properties field, this counts the
    number of distinct entities that could have been in each hash."""
    return(len(set([shape.properties['entity'] for shape in shapes])))
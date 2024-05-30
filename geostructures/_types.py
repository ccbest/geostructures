
from datetime import datetime
from typing import TypeVar, Union


SHAPE_TYPE = TypeVar('_SHAPE_TYPE', bound='BaseShape')
MULTI_SHAPE_TYPE = TypeVar('_MULTI_SHAPE_TYPE', bound='MultiShapeBase')
GEOTIME_TYPE = Union[datetime, 'TimeInterval']

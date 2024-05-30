
from geostructures import Coordinate
from geostructures.utils.functions import round_half_up
from geostructures._base import ShapeLike


def assert_shape_equivalence(shape1: ShapeLike, shape2: ShapeLike, precision: int = 7):
    shape1_coords = [
        Coordinate(round_half_up(x.longitude, precision), round_half_up(x.latitude, precision))
        for x in shape1.bounding_coords()
    ]
    shape2_coords = [
        Coordinate(round_half_up(x.longitude, precision), round_half_up(x.latitude, precision))
        for x in shape2.bounding_coords()
    ]
    assert shape1_coords == shape2_coords
    assert shape1.dt == shape2.dt

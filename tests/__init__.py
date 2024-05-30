
from geostructures import Coordinate
from geostructures.utils.functions import round_half_up
from geostructures._base import BaseShape, ShapeLike, LineLike, PointLike, MultiShapeType


def _assert_shapelike_equivalence(shape1: ShapeLike, shape2: ShapeLike, precision: int = 7):
    shape1_coords = [
        Coordinate(round_half_up(x.longitude, precision), round_half_up(x.latitude, precision))
        for x in shape1.bounding_coords()
    ]
    shape2_coords = [
        Coordinate(round_half_up(x.longitude, precision), round_half_up(x.latitude, precision))
        for x in shape2.bounding_coords()
    ]
    return shape1_coords == shape2_coords and shape1.dt == shape2.dt


def _assert_linelike_equivalence(shape1: LineLike, shape2: LineLike, precision: int = 7):
    shape1_coords = [
        Coordinate(round_half_up(x.longitude, precision), round_half_up(x.latitude, precision))
        for x in shape1.vertices
    ]
    shape2_coords = [
        Coordinate(round_half_up(x.longitude, precision), round_half_up(x.latitude, precision))
        for x in shape2.vertices
    ]
    return shape1_coords == shape2_coords and shape1.dt == shape2.dt


def _assert_pointlike_equivalence(shape1: PointLike, shape2: PointLike, precision: int = 7):
    shape1_coord = Coordinate(
        round_half_up(shape1.centroid.longitude, precision),
        round_half_up(shape1.centroid.latitude, precision)
    )
    shape2_coord = Coordinate(
        round_half_up(shape2.centroid.longitude, precision),
        round_half_up(shape2.centroid.latitude, precision)
    )
    return shape1_coord == shape2_coord and shape1.dt == shape2.dt


def assert_shape_equivalence(shape1: BaseShape, shape2: BaseShape, precision: int = 7):
    if not type(shape1) == type(shape2):
        return False

    if isinstance(shape1, MultiShapeType):
        return all(
            assert_shape_equivalence(x, y, precision)
            for x, y in zip(shape1.geoshapes, shape2.geoshapes)
        )

    if isinstance(shape1, ShapeLike):
        return _assert_shapelike_equivalence(shape1, shape2, precision)

    if isinstance(shape1, LineLike):
        return _assert_linelike_equivalence(shape1, shape2, precision)

    if isinstance(shape1, PointLike):
        return _assert_pointlike_equivalence(shape1, shape2, precision)



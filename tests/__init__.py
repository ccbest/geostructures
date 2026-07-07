
from geostructures import Coordinate
from geostructures.utils.functions import round_half_up
from geostructures.typing import GeoShape, PolygonLike, LineLike, PointLike, MultiShape


def _assert_shapelike_equivalence(shape1: PolygonLike, shape2: PolygonLike, precision: int = 7):
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


def assert_shape_equivalence(shape1: GeoShape, shape2: GeoShape, precision: int = 7):
    """Asserts that two shapes are equivalent to the given precision."""
    assert type(shape1) == type(shape2), f'{type(shape1)} != {type(shape2)}'

    if isinstance(shape1, MultiShape):
        assert len(shape1.geoshapes) == len(shape2.geoshapes), \
            f'{len(shape1.geoshapes)} shapes != {len(shape2.geoshapes)} shapes'
        for x, y in zip(shape1.geoshapes, shape2.geoshapes):
            assert_shape_equivalence(x, y, precision)
        return

    if isinstance(shape1, PolygonLike):
        assert _assert_shapelike_equivalence(shape1, shape2, precision), \
            f'{shape1} not equivalent to {shape2}'
    elif isinstance(shape1, LineLike):
        assert _assert_linelike_equivalence(shape1, shape2, precision), \
            f'{shape1} not equivalent to {shape2}'
    elif isinstance(shape1, PointLike):
        assert _assert_pointlike_equivalence(shape1, shape2, precision), \
            f'{shape1} not equivalent to {shape2}'
    else:
        raise TypeError(f'Unrecognized shape type {type(shape1)}')



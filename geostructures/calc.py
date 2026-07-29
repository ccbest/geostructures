""" Geometric calculations for Coordinates and Geostructures """

__all__ = ['rotate_coordinates']

from typing import List, Tuple, cast

import numpy as np

from geostructures._geometry import ensure_edge_bounds
from geostructures.coordinates import Coordinate


def rotate_coordinates(
        coords: List[Coordinate],
        origin: Coordinate,
        degrees: float,
) -> List[Coordinate]:
    """
    Rotate a set of coordinates around an origin. Returned Coordinate precision will be
    determined by the respective Coordinate's least significant precision.

    Coordinate Z values will be preserved, however M values will not.

    Args:
        coords:
            A list of Coordinates

        origin:
            The centerpoint around which to rotate other Coordinates

        degrees:
            The degrees of rotation to be applied

    Returns:
        List[Coordinate]
    """
    coords = [ensure_edge_bounds(origin, x)[1] for x in coords]
    angle = np.deg2rad(degrees)
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    z = (x.z for x in coords)
    o = np.atleast_2d(origin.to_float()[:2])
    p = np.atleast_2d([x.to_float()[:2] for x in coords])
    return [
        Coordinate(*cast(Tuple[float, float], coord), zval)
        for coord, zval in zip((R @ (p.T - o.T) + o.T).T, z)
    ]

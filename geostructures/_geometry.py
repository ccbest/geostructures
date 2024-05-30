"""
Internal module defining geometric functions used by geostructures
"""

from typing import List

from geostructures.coordinates import Coordinate


def coordinate_vector_cross_product(o: Coordinate, a: Coordinate, b: Coordinate):
    """
    2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.

    Args:
        o: (Coordinate)
            The origin coordinate

        a: (Coordinate)
            The A coordinate, forming the OA vector

        b: (Coordinate)
            The B coordinate, forming the OB vector

    Returns:
        a positive value if OAB makes a counter-clockwise turn, negative for clockwise turn,
        and zero if the points are collinear.
    """
    return (
        (a.longitude - o.longitude) * (b.latitude - o.latitude) -
        (a.latitude - o.latitude) * (b.longitude - o.longitude)
    )


def convex_hull(coordinates: List[Coordinate]) -> List[Coordinate]:
    """
    Computes the convex hull of a set of coordinates using the Monotone Chain algorithm
    (aka Andrew's algorithm). O(n log n) complexity.

    Args:
        coordinates:
            A list of coordinates

    Returns:
        A linear ring representing the convex hull
    """

    # Sort the points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    coordinates = sorted(set(coordinates), key=lambda x: (x.longitude, x.latitude))

    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(coordinates) <= 1:
        return coordinates

    # Build lower hull
    lower: List[Coordinate] = []
    for coord in coordinates:
        while len(lower) >= 2 and coordinate_vector_cross_product(lower[-2], lower[-1], coord) <= 0:
            lower.pop()
        lower.append(coord)

    # Build upper hull
    upper: List[Coordinate] = []
    for coord in reversed(coordinates):
        while len(upper) >= 2 and coordinate_vector_cross_product(upper[-2], upper[-1], coord) <= 0:
            upper.pop()
        upper.append(coord)

    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of the lower list is omitted because it is repeated at
    # the beginning of the other list, but the last point of upper is
    # included because it closes the linear ring
    return lower[:-1] + upper

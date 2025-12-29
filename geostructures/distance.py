# geostructures/calc.py
"""
Geodesic calculation dispatch module.
Supports switching between Haversine (sphere) and Vincenty (ellipsoid) calculations.
"""

__all__ = [
    'haversine_bearing', 'haversine_destination', 'haversine_distance',
    'karney_bearing', 'karney_destination', 'karney_distance',
    'vincenty_bearing', 'vincenty_destination', 'vincenty_distance',
    'bearing_degrees', 'destination_point', 'distance_meters',
]

import math
from typing import Tuple, Literal

from geostructures._const import WGS84_A, WGS84_B, WGS84_F, EARTH_RADIUS_METERS
from geostructures.coordinates import Coordinate


# -------------------------------------------------------------------------
# Haversine Implementation (Spherical)
# -------------------------------------------------------------------------

def haversine_distance(coord1: Coordinate, coord2: Coordinate) -> float:
    """Calculate distance using the Haversine formula (spherical earth)."""
    lon1, lat1 = math.radians(coord1.longitude), math.radians(coord1.latitude)
    lon2, lat2 = math.radians(coord2.longitude), math.radians(coord2.latitude)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return EARTH_RADIUS_METERS * c


def haversine_destination(
        start: Coordinate,
        bearing_degrees: float,
        distance: float,
) -> Coordinate:
    """Calculate destination point using spherical trigonometry."""
    lon1 = math.radians(start.longitude)
    lat1 = math.radians(start.latitude)
    bearing_rad = math.radians(bearing_degrees)

    ang_dist = distance / EARTH_RADIUS_METERS

    lat2 = math.asin(math.sin(lat1) * math.cos(ang_dist) +
                     math.cos(lat1) * math.sin(ang_dist) * math.cos(bearing_rad))

    lon2 = lon1 + math.atan2(math.sin(bearing_rad) * math.sin(ang_dist) * math.cos(lat1),
                             math.cos(ang_dist) - math.sin(lat1) * math.sin(lat2))

    return Coordinate(math.degrees(lon2), math.degrees(lat2))


def haversine_bearing(start: Coordinate, end: Coordinate) -> float:
    """Calculate initial bearing using spherical trigonometry."""
    lon1, lat1 = math.radians(start.longitude), math.radians(start.latitude)
    lon2, lat2 = math.radians(end.longitude), math.radians(end.latitude)

    dlon = lon2 - lon1

    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)

    initial_bearing = math.atan2(y, x)
    return (math.degrees(initial_bearing) + 360) % 360


# -------------------------------------------------------------------------
# Vincenty Implementation (Ellipsoidal)
# -------------------------------------------------------------------------

def vincenty_distance(coord1: Coordinate, coord2: Coordinate) -> float:
    """
    Calculate distance using Vincenty's inverse formula (WGS84 ellipsoid).
    Falls back to Haversine if points are coincident or convergence fails.
    """
    if coord1 == coord2:
        return 0.0

    lon1, lat1 = math.radians(coord1.longitude), math.radians(coord1.latitude)
    lon2, lat2 = math.radians(coord2.longitude), math.radians(coord2.latitude)

    U1 = math.atan((1 - WGS84_F) * math.tan(lat1))
    U2 = math.atan((1 - WGS84_F) * math.tan(lat2))
    L = lon2 - lon1
    Lambda = L

    sinU1, cosU1 = math.sin(U1), math.cos(U1)
    sinU2, cosU2 = math.sin(U2), math.cos(U2)

    MAX_ITER = 200
    for _ in range(MAX_ITER):
        sinLambda, cosLambda = math.sin(Lambda), math.cos(Lambda)
        sinSigma = math.sqrt((cosU2 * sinLambda) ** 2 +
                             (cosU1 * sinU2 - sinU1 * cosU2 * cosLambda) ** 2)

        if sinSigma == 0:
            return 0.0  # Coincident points

        cosSigma = sinU1 * sinU2 + cosU1 * cosU2 * cosLambda
        sigma = math.atan2(sinSigma, cosSigma)
        sinAlpha = cosU1 * cosU2 * sinLambda / sinSigma
        cosSqAlpha = 1 - sinAlpha ** 2

        try:
            cos2SigmaM = cosSigma - 2 * sinU1 * sinU2 / cosSqAlpha
        except ZeroDivisionError:
            cos2SigmaM = 0

        C = WGS84_F / 16 * cosSqAlpha * (4 + WGS84_F * (4 - 3 * cosSqAlpha))
        Lambda_prev = Lambda
        Lambda = L + (1 - C) * WGS84_F * sinAlpha * (
                sigma + C * sinSigma * (cos2SigmaM + C * cosSigma * (-1 + 2 * cos2SigmaM ** 2))
        )

        if abs(Lambda - Lambda_prev) < 1e-12:
            break
    else:
        # Convergence failure (usually antipodal points); fall back to Haversine
        return haversine_distance(coord1, coord2)

    uSq = cosSqAlpha * (WGS84_A ** 2 - WGS84_B ** 2) / (WGS84_B ** 2)
    A = 1 + uSq / 16384 * (4096 + uSq * (-768 + uSq * (320 - 175 * uSq)))
    B = uSq / 1024 * (256 + uSq * (-128 + uSq * (74 - 47 * uSq)))
    deltaSigma = B * sinSigma * (
            cos2SigmaM + B / 4 * (
            cosSigma * (-1 + 2 * cos2SigmaM ** 2) -
            B / 6 * cos2SigmaM * (-3 + 4 * sinSigma ** 2) * (-3 + 4 * cos2SigmaM ** 2)
    )
    )

    return WGS84_B * A * (sigma - deltaSigma)


def vincenty_destination(start: Coordinate, bearing_degrees: float, distance: float) -> Coordinate:
    """Calculate destination using Vincenty's direct formula."""
    if distance == 0:
        return start

    lon1, lat1 = math.radians(start.longitude), math.radians(start.latitude)
    alpha1 = math.radians(bearing_degrees)

    sinAlpha1, cosAlpha1 = math.sin(alpha1), math.cos(alpha1)
    tanU1 = (1 - WGS84_F) * math.tan(lat1)
    cosU1 = 1 / math.sqrt(1 + tanU1 ** 2)
    sinU1 = tanU1 * cosU1

    sigma1 = math.atan2(tanU1, cosAlpha1)
    sinAlpha = cosU1 * sinAlpha1
    cosSqAlpha = 1 - sinAlpha ** 2
    uSq = cosSqAlpha * (WGS84_A ** 2 - WGS84_B ** 2) / (WGS84_B ** 2)

    A = 1 + uSq / 16384 * (4096 + uSq * (-768 + uSq * (320 - 175 * uSq)))
    B = uSq / 1024 * (256 + uSq * (-128 + uSq * (74 - 47 * uSq)))

    sigma = distance / (WGS84_B * A)

    MAX_ITER = 200
    for _ in range(MAX_ITER):
        cos2SigmaM = math.cos(2 * sigma1 + sigma)
        sinSigma, cosSigma = math.sin(sigma), math.cos(sigma)
        deltaSigma = B * sinSigma * (
                cos2SigmaM + B / 4 * (
                cosSigma * (-1 + 2 * cos2SigmaM ** 2) -
                B / 6 * cos2SigmaM * (-3 + 4 * sinSigma ** 2) * (-3 + 4 * cos2SigmaM ** 2)
        )
        )
        sigma_prev = sigma
        sigma = distance / (WGS84_B * A) + deltaSigma
        if abs(sigma - sigma_prev) < 1e-12:
            break

    sinSigma, cosSigma = math.sin(sigma), math.cos(sigma)
    cos2SigmaM = math.cos(2 * sigma1 + sigma)

    # Calculate coordinates
    tmp = sinU1 * sinSigma - cosU1 * cosSigma * cosAlpha1
    lat2 = math.atan2(
        sinU1 * cosSigma + cosU1 * sinSigma * cosAlpha1,
        (1 - WGS84_F) * math.sqrt(sinAlpha ** 2 + tmp ** 2)
    )
    lambda_val = math.atan2(
        sinSigma * sinAlpha1,
        cosU1 * cosSigma - sinU1 * sinSigma * cosAlpha1
    )
    C = WGS84_F / 16 * cosSqAlpha * (4 + WGS84_F * (4 - 3 * cosSqAlpha))
    L = lambda_val - (1 - C) * WGS84_F * sinAlpha * (
            sigma + C * sinSigma * (cos2SigmaM + C * cosSigma * (-1 + 2 * cos2SigmaM ** 2))
    )
    lon2 = lon1 + L

    return Coordinate(math.degrees(lon2), math.degrees(lat2))


def vincenty_bearing(start: Coordinate, end: Coordinate) -> float:
    """
    Calculate the initial bearing (forward azimuth) using Vincenty's inverse formula.

    Args:
        start: The starting Coordinate
        end: The ending Coordinate

    Returns:
        float: Bearing in degrees (0-360)
    """
    if start == end:
        return 0.0

    lon1, lat1 = math.radians(start.longitude), math.radians(start.latitude)
    lon2, lat2 = math.radians(end.longitude), math.radians(end.latitude)

    U1 = math.atan((1 - WGS84_F) * math.tan(lat1))
    U2 = math.atan((1 - WGS84_F) * math.tan(lat2))
    L = lon2 - lon1
    Lambda = L

    sinU1, cosU1 = math.sin(U1), math.cos(U1)
    sinU2, cosU2 = math.sin(U2), math.cos(U2)

    # -------------------------------------------------------------------------
    # Iterative Convergence (Identical to Distance Calculation)
    # -------------------------------------------------------------------------
    MAX_ITER = 200
    for _ in range(MAX_ITER):
        sinLambda, cosLambda = math.sin(Lambda), math.cos(Lambda)

        # eq. 14
        sinSigma = math.sqrt((cosU2 * sinLambda) ** 2 +
                             (cosU1 * sinU2 - sinU1 * cosU2 * cosLambda) ** 2)

        if sinSigma == 0:
            return 0.0  # Coincident points

        # eq. 15
        cosSigma = sinU1 * sinU2 + cosU1 * cosU2 * cosLambda

        # eq. 16
        sigma = math.atan2(sinSigma, cosSigma)

        # eq. 17
        sinAlpha = cosU1 * cosU2 * sinLambda / sinSigma
        cosSqAlpha = 1 - sinAlpha ** 2

        # eq. 18
        try:
            cos2SigmaM = cosSigma - 2 * sinU1 * sinU2 / cosSqAlpha
        except ZeroDivisionError:
            cos2SigmaM = 0

        # eq. 10
        C = WGS84_F / 16 * cosSqAlpha * (4 + WGS84_F * (4 - 3 * cosSqAlpha))

        Lambda_prev = Lambda

        # eq. 11
        Lambda = L + (1 - C) * WGS84_F * sinAlpha * (
                sigma + C * sinSigma * (cos2SigmaM + C * cosSigma * (-1 + 2 * cos2SigmaM ** 2))
        )

        if abs(Lambda - Lambda_prev) < 1e-12:
            break
    else:
        # Convergence failure (usually antipodal points); fall back to Haversine
        return haversine_bearing(start, end)

    # -------------------------------------------------------------------------
    # Azimuth Calculation
    # -------------------------------------------------------------------------
    # Once Lambda has converged, we calculate the forward azimuth (alpha1)
    # eq. 20
    numerator = cosU2 * math.sin(Lambda)
    denominator = cosU1 * sinU2 - sinU1 * cosU2 * math.cos(Lambda)

    alpha1 = math.atan2(numerator, denominator)

    return (math.degrees(alpha1) + 360) % 360


# -------------------------------------------------------------------------
# Karney Implementation (Ellipsoidal)
# -------------------------------------------------------------------------

def karney_distance(coord1: Coordinate, coord2: Coordinate) -> float:
    """
    Calculate distance using Karney's algorithm (via geographiclib).
    Robust against antipodal points and convergence failures.
    """
    from geographiclib.geodesic import Geodesic

    # Inverse returns a dict with 's12' (distance in meters), 'azi1', etc.
    res = Geodesic.WGS84.Inverse(
        coord1.latitude, coord1.longitude,
        coord2.latitude, coord2.longitude
    )
    return res['s12']


def karney_destination(start: Coordinate, bearing_degrees: float, distance: float) -> Coordinate:
    """
    Calculate destination using Karney's algorithm (via geographiclib).
    """
    from geographiclib.geodesic import Geodesic

    # Direct takes (lat1, lon1, azi1, s12)
    res = Geodesic.WGS84.Direct(
        start.latitude, start.longitude,
        bearing_degrees, distance
    )

    return Coordinate(res['lon2'], res['lat2'])


def karney_bearing(start: Coordinate, end: Coordinate) -> float:
    """
    Calculate initial bearing using Karney's algorithm (via geographiclib).
    """
    from geographiclib.geodesic import Geodesic

    res = Geodesic.WGS84.Inverse(
        start.latitude, start.longitude,
        end.latitude, end.longitude
    )

    # geographiclib returns azimuth in range [-180, 180]; normalize to [0, 360]
    return (res['azi1'] + 360) % 360


# -------------------------------------------------------------------------
# Dynamic Dispatch & Configuration
# -------------------------------------------------------------------------

# These declare the distance algo in use (default haversine)
distance_meters = haversine_distance
destination_point = haversine_destination
bearing_degrees = haversine_bearing


_ALGORITHMS = {
    'haversine': (
        haversine_distance,
        haversine_destination,
        haversine_bearing
    ),
    'vincenty': (
        vincenty_distance,
        vincenty_destination,
        vincenty_bearing
    ),
    'karney': (
        karney_distance,
        karney_destination,
        karney_bearing
    )
}


def set_geodesic_algorithm(algorithm: Literal['haversine', 'vincenty', 'karney']):
    """
    Set the global geodesic calculation method.

    Args:
        algorithm: 'haversine' or 'vincenty'
    """
    global distance_meters, destination_point, bearing_degrees

    if algorithm not in _ALGORITHMS:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Options: {list(_ALGORITHMS.keys())}")

    funcs = _ALGORITHMS[algorithm]
    distance_meters = funcs[0]
    destination_point = funcs[1]
    bearing_degrees = funcs[2]
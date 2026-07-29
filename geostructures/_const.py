"""
Constants declarations for geostructures
"""

# WGS84 Ellipsoid Constants
WGS84_A = 6378137.0  # Major axis (meters)
WGS84_F = 1 / 298.257223563  # Flattening
WGS84_B = (1 - WGS84_F) * WGS84_A

# Mean Earth Radius (approximate for Haversine)
EARTH_RADIUS_METERS = 6_371_000.0

# How near two coordinates have to be, in degrees, for geometric predicates to treat
# them as the same point - roughly 11 centimeters at the equator. This matches the six
# decimal places coordinates are normalized to when converted from other formats.
COORDINATE_TOLERANCE = 1e-6

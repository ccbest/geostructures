"""
Breaks the cost of constructing one GeoPoint into its layers, to show where
the per-object time goes (Coordinate, TimeInterval, pydantic validation).

Usage:
    python benchmarks/bench_geopoint_build.py
"""

import sys
import timeit
from datetime import datetime, timezone

from geostructures import Coordinate, GeoPoint
from geostructures.time import TimeInterval


N = 200_000
DT = datetime(2024, 1, 1, tzinfo=timezone.utc)
COORD = Coordinate(-75.0, 40.0)
INTERVAL = TimeInterval(DT, DT)
# The undecorated __init__, if pydantic's validate_call wraps it
RAW_INIT = getattr(GeoPoint.__init__, '__wrapped__', None)


def per_call_us(stmt):
    return min(timeit.repeat(stmt, number=N, repeat=3, globals=globals())) / N * 1e6


def main():
    rows = [
        ('tuple (lon, lat, dt)', '(-75.0, 40.0, DT)'),
        ('Coordinate(x, y)', 'Coordinate(-75.0, 40.0)'),
        ('TimeInterval(d, d)', 'TimeInterval(DT, DT)'),
        ('GeoPoint(c), no dt', 'GeoPoint(COORD)'),
        ('GeoPoint(c, dt=TimeInterval)', 'GeoPoint(COORD, dt=INTERVAL)'),
        ('GeoPoint(c, dt=datetime)', 'GeoPoint(COORD, dt=DT)'),
        ('GeoPoint(Coordinate(..), dt=d)', 'GeoPoint(Coordinate(-75.0, 40.0), dt=DT)'),
    ]
    if RAW_INIT is not None:
        rows.append((
            '  same, pydantic bypassed',
            'p = GeoPoint.__new__(GeoPoint); RAW_INIT(p, Coordinate(-75.0, 40.0), dt=DT)'
        ))

    for label, stmt in rows:
        print(f'{label:<34}{per_call_us(stmt):>8.2f} us')
    print(f'{"sys.getsizeof(GeoPoint.__dict__)":<34}{sys.getsizeof(GeoPoint(COORD, dt=DT).__dict__):>8} B')


if __name__ == '__main__':
    main()

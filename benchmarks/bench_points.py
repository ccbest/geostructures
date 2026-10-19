"""
Compares FeatureCollection (one GeoPoint object per point) with the
array-backed PointCollection on the same N random, time-stamped points.

Usage:
    python benchmarks/bench_points.py [N]        (default 1,000,000)

Each operation is checked to return the same number of points from both
collections. Memory is measured with tracemalloc in a separate pass, because
tracemalloc slows allocation down enough to distort the timings.
"""

import gc
import sys
import time
import tracemalloc
from datetime import datetime, timedelta, timezone

import numpy as np

from geostructures import Coordinate, FeatureCollection, GeoBox, GeoCircle, GeoPoint, GeoPolygon, PointCollection
from geostructures.time import TimeInterval


def timed(fn):
    gc.collect()
    start = time.perf_counter()
    result = fn()
    return time.perf_counter() - start, result


def bytes_per_point(fn, n):
    gc.collect()
    tracemalloc.start()
    result = fn()
    used = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    del result
    return used / n


def main(n):
    rng = np.random.default_rng(0)
    lon = rng.uniform(-80, -70, n)
    lat = rng.uniform(35, 45, n)
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    secs = rng.integers(0, 86400 * 30, n)

    # Inputs each approach would naturally start from (not timed)
    datetimes = [t0 + timedelta(seconds=int(s)) for s in secs]
    times64 = np.datetime64('2024-01-01T00:00:00', 'us') + secs.astype('timedelta64[s]')

    def build_objects():
        return FeatureCollection([GeoPoint(Coordinate(x, y), dt=d) for x, y, d in zip(lon, lat, datetimes)])

    def build_arrays():
        return PointCollection(lon, lat, dt_start=times64)

    box = GeoBox(Coordinate(-76, 41), Coordinate(-74, 39))
    circle = GeoCircle(Coordinate(-75, 40), 100_000)
    polygon = GeoPolygon([
        Coordinate(-77, 38), Coordinate(-73, 38.5), Coordinate(-74, 42),
        Coordinate(-75.5, 40), Coordinate(-77, 41), Coordinate(-77, 38)
    ])
    window = TimeInterval(t0 + timedelta(days=10), t0 + timedelta(days=12))

    rows = []
    t_obj, fc = timed(build_objects)
    t_arr, pc = timed(build_arrays)
    rows.append(('build', t_obj, t_arr, len(fc), len(pc)))

    t_conv, converted = timed(lambda: PointCollection.from_points(fc))
    assert len(converted) == n

    operations = [
        ('bounds', lambda c: c.bounds),
        ('filter_by_dt', lambda c: c.filter_by_dt(window)),
        ('in GeoBox', lambda c: c.filter_contained_by(box)),
        ('in GeoCircle', lambda c: c.filter_contained_by(circle)),
        ('in GeoPolygon', lambda c: c.filter_contained_by(polygon)),
    ]
    for label, op in operations:
        t_obj, r_obj = timed(lambda: op(fc))
        t_arr, r_arr = timed(lambda: op(pc))
        if isinstance(r_obj, tuple):
            assert r_obj == r_arr, label
            rows.append((label, t_obj, t_arr, None, None))
        else:
            assert len(r_obj) == len(r_arr), label
            rows.append((label, t_obj, t_arr, len(r_obj), len(r_arr)))

    t_mat, _ = timed(lambda: pc[:100_000].to_points())

    # Release the timed collections before measuring memory
    fc = pc = converted = None
    mem_obj = bytes_per_point(build_objects, n)
    mem_arr = bytes_per_point(build_arrays, n)

    print(f'N = {n:,} points')
    print(f'{"operation":<16}{"FeatureCollection":>19}{"PointCollection":>17}{"speedup":>10}{"matches":>12}')
    for label, t_obj, t_arr, c_obj, c_arr in rows:
        matches = '' if c_obj is None else f'{c_arr:,}'
        print(f'{label:<16}{t_obj:>18.3f}s{t_arr:>16.4f}s{t_obj / t_arr:>9.0f}x{matches:>12}')
    print(f'{"memory / point":<16}{mem_obj:>18,.0f}B{mem_arr:>16,.0f}B{mem_obj / mem_arr:>9.0f}x')
    print()
    print(f'PointCollection.from_points(FeatureCollection): {t_conv:.2f}s')
    print(f'Materializing 100,000 GeoPoints from a PointCollection: {t_mat:.2f}s')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000)

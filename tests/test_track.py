
from datetime import datetime, date, timezone

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from geostructures.coordinates import Coordinate
from geostructures.structures import GeoBox, GeoPoint, GeoPolygon
from geostructures.time import DateInterval, TimeInterval
from geostructures.collections import Track



def test_track_add():
    track1 = Track([
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 1)),
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 2)),
    ])
    track2 = Track([
        GeoPoint(Coordinate(2.0, 2.0), datetime(2020, 1, 1)),
        GeoPoint(Coordinate(2.0, 2.0), datetime(2020, 1, 2)),
    ])
    track_new = track1 + track2
    assert all(x in track_new.geoshapes for x in track1)
    assert all(x in track_new.geoshapes for x in track2)

    # Make sure nothing mutated
    assert all([x not in track2.geoshapes for x in track1])
    assert all([x not in track1.geoshapes for x in track2])


def test_track_contains():
    track1 = Track([
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 1)),
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 2)),
    ])
    assert GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 1)) in track1

    assert GeoPoint(Coordinate(2.0, 2.0), datetime(2020, 1, 1)) not in track1
    assert GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 3)) not in track1


def test_track_bool():
    track1 = Track([
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 1)),
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 2)),
    ])
    assert track1

    new_track = Track([])
    assert not new_track


def test_track_getitem():
    track1 = Track([
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 1)),
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 2)),
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 3)),
    ])

    assert track1[:datetime(2020, 1, 2)] == Track([GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 1))])
    assert track1[datetime(2020, 1, 2):] == Track([
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 2)),
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 3))
    ])

    assert track1[datetime(2020, 1, 2):datetime(2020, 1, 3)] == Track(
        [GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 2))]
    )


def test_track_iter():
    assert list(Track([
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 1)),
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 2)),
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 3)),
    ])) == [
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 1)),
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 2)),
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 3)),
    ]


def test_track_eq():
    track1 = Track([
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 1)),
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 2)),
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 3)),
    ])
    assert track1 == Track([
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 1)),
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 2)),
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 3)),
    ])
    assert track1 != 'not a track'
    assert track1 != Track([
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 1)),
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 2)),
    ])


def test_track_len():
    track1 = Track([
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 1)),
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 2)),
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 3)),
    ])
    assert len(track1) == 3
    track1.geoshapes.pop(0)
    assert len(track1) == 2


def test_track_copy():
    track1 = Track([
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 1)),
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 2)),
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 3)),
    ])
    new_track = track1.copy()
    new_track.geoshapes.pop()
    assert len(new_track) != len(track1)


def test_track_first():
    track1 = Track([
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 1)),
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 2)),
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 3)),
    ])
    assert track1.first == GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 1))
    track1.geoshapes.pop(0)
    assert track1.first == GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 2))

    with pytest.raises(ValueError):
        _ = Track([]).first


def test_track_last():
    track1 = Track([
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 1)),
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 2)),
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 3)),
    ])
    assert track1.last == GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 3))
    track1.geoshapes.pop(-1)
    assert track1.last == GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 2))

    with pytest.raises(ValueError):
        _ = Track([]).last


def test_track_start():
    track1 = Track([
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 1)),
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 2)),
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 3)),
    ])
    assert track1.start == datetime(2020, 1, 1, tzinfo=timezone.utc)
    track1.geoshapes.pop(0)
    assert track1.start == datetime(2020, 1, 2, tzinfo=timezone.utc)

    with pytest.raises(ValueError):
        _ = Track([]).start


def test_track_finish():
    track1 = Track([
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 1)),
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 2)),
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 3)),
    ])
    assert track1.finish == datetime(2020, 1, 3, tzinfo=timezone.utc)
    track1.geoshapes.pop(-1)
    assert track1.finish == datetime(2020, 1, 2, tzinfo=timezone.utc)

    with pytest.raises(ValueError):
        _ = Track([]).finish


def test_track_convex_hull():
    track = Track(
        [
            GeoPoint(Coordinate('-2.0', '-2.0'), datetime(2020, 1, 1)),
            GeoPoint(Coordinate('2.0', '0.0'), datetime(2020, 1, 1, 1)),
            GeoPoint(Coordinate('3.0', '3.0'), datetime(2020, 1, 1, 2)),
            GeoPoint(Coordinate('2.0', '6.0'), datetime(2020, 1, 1, 3)),
            GeoPoint(Coordinate('-2.0', '8.0'), datetime(2020, 1, 1, 4)),
            GeoPoint(Coordinate('-6.0', '6.0'), datetime(2020, 1, 1, 5)),
            GeoPoint(Coordinate('-7.0', '3.0'), datetime(2020, 1, 1, 6)),
            GeoPoint(Coordinate('-6.0', '0.0'), datetime(2020, 1, 1, 6)),
            GeoPoint(Coordinate('-2.0', '-2.0'), datetime(2020, 1, 1, 7)),
            GeoPoint(Coordinate('-2.0', '4.0'), datetime(2020, 1, 1, 8)),
        ]
    )
    points = [x.centroid.to_float() for x in track.geoshapes]
    hull = ConvexHull(points)
    assert GeoPolygon([Coordinate(*points[x]) for x in hull.vertices]) == track.convex_hull
    
    # Fewer than 3 points
    new_track = Track([
        GeoPoint(Coordinate('-2.0', '-2.0'), datetime(2020, 1, 1)),
        GeoPoint(Coordinate('-2.0', '-2.0'), datetime(2020, 1, 2))
    ])
    with pytest.raises(ValueError):
        _ = new_track.convex_hull


def test_track_distances():
    track1 = Track(
        [
            GeoPoint(Coordinate('0.0000', '1.0000'), datetime(2020, 1, 1)),
            GeoPoint(Coordinate('0.0010', '1.0010'), datetime(2020, 1, 1, 1)),
            GeoPoint(Coordinate('0.0020', '1.0020'), datetime(2020, 1, 1, 2)),
            GeoPoint(Coordinate('0.0030', '1.0030'), datetime(2020, 1, 1, 3)),
            GeoPoint(Coordinate('0.0040', '1.0040'), datetime(2020, 1, 1, 4)),
            GeoPoint(Coordinate('0.0050', '1.0050'), datetime(2020, 1, 1, 5)),
            GeoPoint(Coordinate('0.0060', '1.0060'), datetime(2020, 1, 1, 6)),
            GeoPoint(Coordinate('0.0070', '1.0070'), datetime(2020, 1, 1, 7)),
        ]
    )
    np.testing.assert_array_equal(
        np.round(track1.centroid_distances, 3),
        np.round(np.array(
            [
                157.241, 157.241, 157.241, 157.241, 157.241,
                157.241, 157.241
            ]
        ), 3)
    )

    with pytest.raises(ValueError):
        _ = Track([GeoPoint(Coordinate('0.0000', '1.0000'), datetime(2020, 1, 1))]).centroid_distances


def test_track_time_diffs():
    track1 = Track(
        [
            GeoPoint(Coordinate('0.0000', '1.0000'), datetime(2020, 1, 1)),
            GeoPoint(Coordinate('0.0010', '1.0010'), datetime(2020, 1, 1, 1)),
            GeoPoint(Coordinate('0.0020', '1.0020'), datetime(2020, 1, 1, 2)),
            GeoPoint(Coordinate('0.0030', '1.0030'), datetime(2020, 1, 1, 3)),
            GeoPoint(Coordinate('0.0040', '1.0040'), datetime(2020, 1, 1, 4)),
            GeoPoint(Coordinate('0.0050', '1.0050'), datetime(2020, 1, 1, 5)),
            GeoPoint(Coordinate('0.0060', '1.0060'), datetime(2020, 1, 1, 6)),
            GeoPoint(Coordinate('0.0070', '1.0070'), datetime(2020, 1, 1, 7)),
        ]
    )

    assert track1.time_start_diffs[0] == track1.geoshapes[1].dt - track1.geoshapes[0].dt
    assert len(track1) - 1 == len(track1.time_start_diffs)


def test_track_has_duplicates():
    track = Track(
        [
            GeoPoint(Coordinate('0.0000', '1.0000'), datetime(2020, 1, 1)),
            GeoPoint(Coordinate('1.0000', '0.0000'), datetime(2020, 1, 1)),
            GeoPoint(Coordinate('0.0010', '1.0010'), datetime(2020, 1, 1, 1))
        ]
    )
    assert track.has_duplicate_timestamps

    track = Track(
        [
            GeoPoint(Coordinate('0.0000', '1.0000'), datetime(2020, 1, 1)),
            GeoPoint(Coordinate('1.0000', '0.0000'), datetime(2020, 1, 1, 0, 1)),
            GeoPoint(Coordinate('0.0010', '1.0010'), datetime(2020, 1, 1, 1))
        ]
    )
    assert not track.has_duplicate_timestamps


def test_track_convolve_duplicate_timestamps():
    track = Track(
        [
            GeoPoint(Coordinate('0.0000', '1.0000'), datetime(2020, 1, 1)),
            GeoPoint(Coordinate('1.0000', '0.0000'), datetime(2020, 1, 1)),
            GeoPoint(Coordinate('0.0010', '1.0010'), datetime(2020, 1, 1, 1))
        ]
    )
    expected = Track(
        [
            GeoPoint(Coordinate('0.5000', '0.5000'), datetime(2020, 1, 1)),
            GeoPoint(Coordinate('0.0010', '1.0010'), datetime(2020, 1, 1, 1))
        ]
    )
    assert track.convolve_duplicate_timestamps() == expected


def test_track_intersects():
    track1 = Track(
        [
            GeoPoint(Coordinate('0.0000', '1.0000'), datetime(2020, 1, 1)),
            GeoPoint(Coordinate('0.0010', '1.0010'), datetime(2020, 1, 1, 1)),
            GeoPoint(Coordinate('0.0020', '1.0020'), datetime(2020, 1, 1, 2)),
            GeoPoint(Coordinate('0.0030', '1.0030'), datetime(2020, 1, 1, 3)),
            GeoPoint(Coordinate('0.0040', '1.0040'), datetime(2020, 1, 1, 4)),
            GeoPoint(Coordinate('0.0050', '1.0050'), datetime(2020, 1, 1, 5)),
            GeoPoint(Coordinate('0.0060', '1.0060'), datetime(2020, 1, 1, 6)),
            GeoPoint(Coordinate('0.0070', '1.0070'), datetime(2020, 1, 1, 7)),
        ]
    )
    gbox = GeoBox(Coordinate(0., 2.), Coordinate(2., 0.))
    assert track1.intersects(gbox)

    gbox = GeoBox(Coordinate(5., 6.), Coordinate(6., 5.))
    assert not track1.intersects(gbox)

    gbox = GeoBox(Coordinate(0., 2.), Coordinate(2., 0.), dt=datetime(2020, 1, 1))
    assert track1.intersects(gbox)
    gbox = GeoBox(Coordinate(0., 2.), Coordinate(2., 0.), dt=datetime(2019, 1, 1))
    assert not track1.intersects(gbox)

    gbox = GeoBox(Coordinate(0., 2.), Coordinate(2., 0.), dt=date(2020, 1, 1))
    assert track1.intersects(gbox)
    gbox = GeoBox(Coordinate(0., 2.), Coordinate(2., 0.), dt=date(2019, 1, 1))
    assert not track1.intersects(gbox)

    gbox = GeoBox(Coordinate(0., 2.), Coordinate(2., 0.), dt=datetime(2020, 1, 1))
    assert track1.intersects(gbox)
    gbox = GeoBox(Coordinate(0., 2.), Coordinate(2., 0.), dt=datetime(2019, 1, 1))
    assert not track1.intersects(gbox)

    gbox = GeoBox(Coordinate(0., 2.), Coordinate(2., 0.), dt=DateInterval(date(2020, 1, 1), date(2020, 1, 2)))
    assert track1.intersects(gbox)
    gbox = GeoBox(Coordinate(0., 2.), Coordinate(2., 0.),dt=DateInterval(date(2019, 1, 1), date(2019, 1, 2)))
    assert not track1.intersects(gbox)

    gbox = GeoBox(Coordinate(0., 2.), Coordinate(2., 0.), dt=TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 2)))
    assert track1.intersects(gbox)
    gbox = GeoBox(Coordinate(0., 2.), Coordinate(2., 0.),dt=TimeInterval(datetime(2019, 1, 1), datetime(2019, 1, 2)))
    assert not track1.intersects(gbox)


def test_track_intersection():
    track1 = Track(
        [
            GeoPoint(Coordinate('0.0000', '1.0000'), datetime(2020, 1, 1)),
            GeoPoint(Coordinate('0.0010', '1.0010'), datetime(2020, 1, 1, 1)),
            GeoPoint(Coordinate('0.0020', '1.0020'), datetime(2020, 1, 1, 2)),
            GeoPoint(Coordinate('0.0030', '1.0030'), datetime(2020, 1, 1, 3)),
            GeoPoint(Coordinate('0.0040', '1.0040'), datetime(2020, 1, 1, 4)),
            GeoPoint(Coordinate('0.0050', '1.0050'), datetime(2020, 1, 1, 5)),
            GeoPoint(Coordinate('0.0060', '1.0060'), datetime(2020, 1, 1, 6)),
            GeoPoint(Coordinate('0.0070', '1.0070'), datetime(2020, 1, 1, 7)),
        ]
    )

    gbox = GeoBox(Coordinate(0., 2.), Coordinate(2., 0.))
    assert track1.intersection(gbox) == track1

    gbox = GeoBox(Coordinate(0., 1.005), Coordinate(1., 0.))
    assert len(track1.intersection(gbox)) == 6

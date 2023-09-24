
from datetime import datetime, time, timezone

import numpy as np
import pytest
import shapely
from scipy.spatial import ConvexHull

from geostructures.coordinates import Coordinate
from geostructures import GeoBox, GeoLineString, GeoPoint, GeoPolygon
from geostructures.time import TimeInterval
from geostructures.collections import Track, FeatureCollection


def test_collection_bool():
    track1 = Track([
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 1)),
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 2)),
    ])
    assert track1

    new_track = Track([])
    assert not new_track


def test_collection_contains():
    track1 = Track([
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 1)),
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 2)),
    ])
    assert GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 1)) in track1

    assert GeoPoint(Coordinate(2.0, 2.0), datetime(2020, 1, 1)) not in track1
    assert GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 3)) not in track1


def test_collection_iter():
    assert list(Track([
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 1)),
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 2)),
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 3)),
    ])) == [
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 1)),
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 2)),
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 3)),
    ]


def test_collection_centroid():
    track1 = Track([
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 1)),
        GeoPoint(Coordinate(2.0, 2.0), datetime(2020, 1, 2)),
        GeoPoint(Coordinate(3.0, 1.5), datetime(2020, 1, 3)),
    ])
    assert track1.centroid == Coordinate(2.0, 1.5)


def test_collection_len():
    track1 = Track([
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 1)),
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 2)),
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 3)),
    ])
    assert len(track1) == 3
    track1.geoshapes.pop(0)
    assert len(track1) == 2


def test_collection_from_geopandas():
    col = FeatureCollection([
        GeoPolygon(
            [Coordinate(0.0, 1.0), Coordinate(1.0, 1.0), Coordinate(1.0, 0.0)],
            dt=datetime(2020, 1, 1)
        ),
        GeoLineString(
            [Coordinate(0.0, 1.0), Coordinate(1.0, 1.0), Coordinate(1.0, 0.0)],
            dt=TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 2))
        ),
        GeoPoint(
            Coordinate(0.0, 1.0),
            dt=None
        )
    ])
    df = col.to_geopandas()
    new_col = FeatureCollection.from_geopandas(df)

    assert col == new_col


def test_collection_from_shapely():
    gls = GeoLineString([Coordinate(0.0, 0.0), Coordinate(1.0, 1.0), Coordinate(2.0, 2.0)])
    gpolygon = GeoPolygon([Coordinate(0.0, 0.0), Coordinate(1.0, 1.0), Coordinate(2.0, 0.0), Coordinate(0.0, 0.0)])
    gpoint = GeoPoint(Coordinate(0.0, 0.0))
    expected = FeatureCollection([gls, gpolygon, gpoint])

    gcol = shapely.GeometryCollection([x.to_shapely() for x in expected])
    assert FeatureCollection.from_shapely(gcol) == expected


def test_collection_to_geojson():
    shapes = [
        GeoBox(Coordinate(0.0, 1.0), Coordinate(1.0, 0.0)),
        GeoBox(Coordinate(0.0, 2.0), Coordinate(2.0, 0.0)),
    ]
    col = FeatureCollection(shapes)
    assert col.to_geojson() == {
        'type': 'FeatureCollection',
        'features': [
            shapes[0].to_geojson(id=0),
            shapes[1].to_geojson(id=1)
        ]
    }

    # Assert properties end up in right place
    assert col.to_geojson(properties={'test_prop': 1}) == {
        'type': 'FeatureCollection',
        'features': [
            shapes[0].to_geojson(id=0, properties={'test_prop': 1}),
            shapes[1].to_geojson(id=1, properties={'test_prop': 1})
        ]
    }

    # Assert kwargs end up in correct place
    assert col.to_geojson(test_kwarg=1) == {
        'type': 'FeatureCollection',
        'features': [
            shapes[0].to_geojson(id=0, test_kwarg=1),
            shapes[1].to_geojson(id=1, test_kwarg=1),
        ]
    }


def test_collection_convex_hull():
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


def test_featurecollection_add():
    fcol1 = FeatureCollection([
        GeoBox(Coordinate(0.0, 1.0), Coordinate(1.0, 0.0)),
    ])

    fcol2 = FeatureCollection([
        GeoBox(Coordinate(0.0, 2.0), Coordinate(2.0, 0.0)),
    ])

    assert fcol1 + fcol2 == FeatureCollection([
        GeoBox(Coordinate(0.0, 1.0), Coordinate(1.0, 0.0)),
        GeoBox(Coordinate(0.0, 2.0), Coordinate(2.0, 0.0)),
    ])

    with pytest.raises(ValueError):
        _ = fcol1 + 2


def test_featurecollection_eq():
    fcol1 = FeatureCollection([
        GeoBox(Coordinate(0.0, 1.0), Coordinate(1.0, 0.0)),
    ])
    fcol2 = FeatureCollection([
        GeoBox(Coordinate(0.0, 2.0), Coordinate(2.0, 0.0)),
    ])
    assert fcol1 == FeatureCollection([
        GeoBox(Coordinate(0.0, 1.0), Coordinate(1.0, 0.0)),
    ])
    assert fcol1 != fcol2

    assert fcol1 != 2


def test_featurecollection_getitem():
    fcol = FeatureCollection([
        GeoBox(Coordinate(0.0, 1.0), Coordinate(1.0, 0.0)),
        GeoBox(Coordinate(0.0, 2.0), Coordinate(2.0, 0.0)),
        GeoBox(Coordinate(0.0, 3.0), Coordinate(3.0, 0.0)),
    ])
    assert fcol[0] == GeoBox(Coordinate(0.0, 1.0), Coordinate(1.0, 0.0))
    assert fcol[0:2] == [
        GeoBox(Coordinate(0.0, 1.0), Coordinate(1.0, 0.0)),
        GeoBox(Coordinate(0.0, 2.0), Coordinate(2.0, 0.0)),
    ]


def test_featurecollection_iter():
    fcol = FeatureCollection([
        GeoBox(Coordinate(0.0, 1.0), Coordinate(1.0, 0.0)),
        GeoBox(Coordinate(0.0, 2.0), Coordinate(2.0, 0.0)),
        GeoBox(Coordinate(0.0, 3.0), Coordinate(3.0, 0.0)),
    ])
    assert list(fcol) == [
        GeoBox(Coordinate(0.0, 1.0), Coordinate(1.0, 0.0)),
        GeoBox(Coordinate(0.0, 2.0), Coordinate(2.0, 0.0)),
        GeoBox(Coordinate(0.0, 3.0), Coordinate(3.0, 0.0)),
    ]


def test_featurecollection_len():
    fcol = FeatureCollection([
        GeoBox(Coordinate(0.0, 1.0), Coordinate(1.0, 0.0)),
        GeoBox(Coordinate(0.0, 2.0), Coordinate(2.0, 0.0)),
        GeoBox(Coordinate(0.0, 3.0), Coordinate(3.0, 0.0)),
    ])
    assert len(fcol) == 3
    fcol.geoshapes.pop(0)
    assert len(fcol) == 2


def test_featurecollection_repr():
    fcol = FeatureCollection([
        GeoBox(Coordinate(0.0, 1.0), Coordinate(1.0, 0.0)),
        GeoBox(Coordinate(0.0, 2.0), Coordinate(2.0, 0.0)),
        GeoBox(Coordinate(0.0, 3.0), Coordinate(3.0, 0.0)),
    ])
    assert repr(fcol) == '<FeatureCollection with 3 shapes>'

    fcol = FeatureCollection([])
    assert repr(fcol) == '<Empty FeatureCollection>'


def test_featurecollection_copy():
    fcol = FeatureCollection([
        GeoBox(Coordinate(0.0, 1.0), Coordinate(1.0, 0.0)),
        GeoBox(Coordinate(0.0, 2.0), Coordinate(2.0, 0.0)),
        GeoBox(Coordinate(0.0, 3.0), Coordinate(3.0, 0.0)),
    ])
    fcol_copy = fcol.copy()

    # Non-shallow mutation, should still be equal
    fcol_copy[0].dt = datetime(2020, 1, 1)
    assert fcol == fcol_copy

    # Shallow mutation, should no longer be equal
    fcol_copy.geoshapes.pop(0)
    assert fcol != fcol_copy


def test_track_init():
    _ = Track([
        GeoBox(Coordinate(0.0, 1.0), Coordinate(1.0, 0.0), dt=datetime(2020, 1, 1)),
        GeoBox(Coordinate(0.0, 2.0), Coordinate(2.0, 0.0), dt=datetime(2020, 1, 1)),
    ])

    with pytest.raises(ValueError):
        _ = Track([
            GeoBox(Coordinate(0.0, 1.0), Coordinate(1.0, 0.0)),
            GeoBox(Coordinate(0.0, 2.0), Coordinate(2.0, 0.0), dt=datetime(2020, 1, 1)),
        ])


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

    with pytest.raises(ValueError):
        _ = track1 + 2


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


def test_track_getitem():
    track1 = Track([
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 1)),
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 2)),
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 3)),
    ])

    assert track1[datetime(2020, 1, 1)] == Track([GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 1))])

    assert track1[:datetime(2020, 1, 2)] == Track([GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 1))])
    assert track1[datetime(2020, 1, 2):] == Track([
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 2)),
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 3))
    ])

    assert track1[datetime(2020, 1, 2):datetime(2020, 1, 3)] == Track(
        [GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 2))]
    )


def test_track_repr():
    track = Track([
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 1)),
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 2)),
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 3)),
    ])
    assert repr(track) == '<Track with 3 shapes from 2020-01-01T00:00:00+00:00 - 2020-01-03T00:00:00+00:00>'

    track = Track([])
    assert repr(track) == '<Empty Track>'


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


def test_track_speed_diffs():
    track1 = Track([
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 1)),
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 2)),
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 3)),
    ])
    np.testing.assert_array_equal(track1.speed_diffs, np.array([0., 0.]))

    track1 = Track([
        GeoPoint(Coordinate(1.0, 1.0), datetime(2020, 1, 1)),
        GeoPoint(Coordinate(2.0, 2.0), datetime(2020, 1, 2)),
        GeoPoint(Coordinate(3.0, 3.0), datetime(2020, 1, 3)),
    ])
    np.testing.assert_array_equal(
        np.round(track1.speed_diffs, 5),
        np.round(np.array([1.8197388, 1.81918463]), 5)
    )


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

    with pytest.raises(ValueError):
        track1 = Track(
            [
                GeoPoint(Coordinate('0.0000', '1.0000'), datetime(2020, 1, 1))
            ]
        )
        _ = track1.time_start_diffs


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

    track = Track(
        [
            GeoPoint(Coordinate('0.0000', '1.0000'), datetime(2020, 1, 1)),
            GeoPoint(Coordinate('1.0000', '0.0000'), datetime(2020, 1, 1, 1)),
            GeoPoint(Coordinate('0.0010', '1.0010'), datetime(2020, 1, 1, 2))
        ]
    )
    assert track.convolve_duplicate_timestamps() == track


def test_track_filter_by_time():
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
            GeoPoint(Coordinate('0.0070', '1.0070'), TimeInterval(datetime(2020, 1, 1, 1), datetime(2020, 1, 1, 7))),
        ]
    )
    assert track1.filter_by_time(time(2, 30), time(5, 30)) == Track([
        GeoPoint(Coordinate('0.0030', '1.0030'), datetime(2020, 1, 1, 3)),
        GeoPoint(Coordinate('0.0040', '1.0040'), datetime(2020, 1, 1, 4)),
        GeoPoint(Coordinate('0.0050', '1.0050'), datetime(2020, 1, 1, 5)),
        GeoPoint(Coordinate('0.0070', '1.0070'), TimeInterval(datetime(2020, 1, 1, 1), datetime(2020, 1, 1, 7))),
    ])


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

    # dt is Datetime
    gbox = GeoBox(Coordinate(0., 2.), Coordinate(2., 0.), dt=datetime(2020, 1, 1))
    assert track1.intersects(gbox)
    gbox = GeoBox(Coordinate(0., 2.), Coordinate(2., 0.), dt=datetime(2019, 1, 1))
    assert not track1.intersects(gbox)

    gbox = GeoBox(Coordinate(0., 2.), Coordinate(2., 0.), dt=datetime(2020, 1, 1))
    assert track1.intersects(gbox)
    gbox = GeoBox(Coordinate(0., 2.), Coordinate(2., 0.), dt=datetime(2019, 1, 1))
    assert not track1.intersects(gbox)

    # dt is TimeInterval
    gbox = GeoBox(Coordinate(0., 2.), Coordinate(2., 0.), dt=TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 2)))
    assert track1.intersects(gbox)
    gbox = GeoBox(Coordinate(0., 2.), Coordinate(2., 0.), dt=TimeInterval(datetime(2019, 1, 1), datetime(2019, 1, 2)))
    assert not track1.intersects(gbox)

    with pytest.raises(ValueError):
        _ = GeoBox(Coordinate(0., 2.), Coordinate(2., 0.), dt='not a datetime')
        track1.intersects(_)


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

    gbox = GeoBox(Coordinate(0., 2.), Coordinate(2., 0.), dt=datetime(2020, 1, 1, 3))
    assert len(track1.intersection(gbox)) == 1


from datetime import datetime, time, timezone
import os
import tempfile
from zipfile import ZipFile

import numpy as np
import pytest
import shapely

from geostructures.coordinates import Coordinate
from geostructures.multistructures import *
from geostructures import GeoBox, GeoCircle, GeoLineString, GeoPoint, GeoPolygon, GeoRing, MultiGeoPoint
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


def test_collection_bounds():
    col1 = FeatureCollection([
        GeoBox(Coordinate(-1., 0.), Coordinate(0., -5)),
        GeoBox(Coordinate(-0.5, 2.), Coordinate(2., -7)),
    ])
    assert col1.bounds == (-1, -7, 2, 2)


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


def test_collection_from_fastkml_folder():
    from fastkml import Folder
    folder = Folder(name='test folder')
    folder.append(
        GeoPoint(
            Coordinate(1., 0.),
            dt=datetime(2020, 1, 1),
            properties={'test': 'prop'}
        ).to_fastkml_placemark(
            description='test description',
            name='test name'
        )
    )
    actual = FeatureCollection.from_fastkml_folder(folder)
    expected = FeatureCollection([
        GeoPoint(
            Coordinate(1., 0.),
            dt=datetime(2020, 1, 1),
            properties={'test': 'prop'}
        )
    ])
    assert actual == expected


def test_collection_filter_by_dt():
    col = FeatureCollection([
        # Intersects in point in time
        GeoCircle(
            Coordinate(0.0, 1.0),
            500,
            dt=datetime(2020, 1, 1)
        ),
        # Intersects in timespan
        GeoCircle(
            Coordinate(0.0, 1.0),
            500,
            dt=TimeInterval(datetime(2019, 12, 1), datetime(2020, 1, 2))
        ),
        # Does not intersect in space
        GeoPoint(
            Coordinate(5.0, 5.0),
            dt=None
        ),
        # Does not intersect in time
        GeoPoint(
            Coordinate(0.0, 0.0),
            dt=datetime(2020, 1, 5)
        ),
        # Intersects in space, time eternal
        GeoPoint(
            Coordinate(0.0, 0.0),
            dt=None
        ),
    ])
    test_interval = TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 2))
    assert col.filter_by_dt(test_interval) == FeatureCollection([
        # Intersects in point in time
        GeoCircle(
            Coordinate(0.0, 1.0),
            500,
            dt=datetime(2020, 1, 1)
        ),
        # Intersects in timespan
        GeoCircle(
            Coordinate(0.0, 1.0),
            500,
            dt=TimeInterval(datetime(2019, 12, 1), datetime(2020, 1, 2))
        ),
    ])

    assert col.filter_by_dt(datetime(2020, 1, 1)) == FeatureCollection([
        GeoCircle(
            Coordinate(0.0, 1.0),
            500,
            dt=datetime(2020, 1, 1)
        ),
    ])


def test_collection_filter_contained_by():
    col = FeatureCollection([
        GeoPoint(Coordinate(-1, 0)),
        GeoPoint(Coordinate(0, 0)),
        GeoPoint(Coordinate(1, 0)),
    ])
    expected = FeatureCollection([GeoPoint(Coordinate(0, 0))])
    actual = col.filter_contained_by(GeoCircle(Coordinate(0., 0.), 100))
    assert actual == expected


def test_collection_filter_contains():
    col = FeatureCollection([
        GeoCircle(Coordinate(-1, 0), 100),
        GeoCircle(Coordinate(0, 0), 100),
        GeoCircle(Coordinate(1, 0), 100),
    ])
    expected = FeatureCollection([GeoCircle(Coordinate(0, 0), 100)])
    actual = col.filter_contains(GeoPoint(Coordinate(0., 0.)))
    assert actual == expected


def test_collection_filter_by_intersection():
    col = FeatureCollection([
        # Intersects in point in time
        GeoCircle(
            Coordinate(0.0, 1.0),
            500,
            dt=datetime(2020, 1, 1)
        ),
        # Intersects in timespan
        GeoCircle(
            Coordinate(0.0, 1.0),
            500,
            dt=TimeInterval(datetime(2019, 12, 1), datetime(2020, 1, 2))
        ),
        # Does not intersect in space
        GeoPoint(
            Coordinate(5.0, 5.0),
            dt=None
        ),
        # Does not intersect in time
        GeoPoint(
            Coordinate(0.0, 0.0),
            dt=datetime(2020, 1, 5)
        ),
        # Intersects in space, time eternal
        GeoPoint(
            Coordinate(0.0, 1.0),
            dt=None
        ),
    ])

    intersecting_shape = GeoCircle(Coordinate(0.0, 1.0), 100)
    assert col.filter_by_intersection(intersecting_shape) == FeatureCollection([
        # Intersects in point in time
        GeoCircle(
            Coordinate(0.0, 1.0),
            500,
            dt=datetime(2020, 1, 1)
        ),
        # Intersects in timespan
        GeoCircle(
            Coordinate(0.0, 1.0),
            500,
            dt=TimeInterval(datetime(2019, 12, 1), datetime(2020, 1, 2))
        ),
        # Intersects in space, time eternal
        GeoPoint(
            Coordinate(0.0, 1.0),
            dt=None
        ),
    ])

    test_interval = TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 2))
    intersecting_shape = GeoCircle(Coordinate(0.0, 1.0), 100, dt=test_interval)
    assert col.filter_by_intersection(intersecting_shape) == FeatureCollection([
        # Intersects in point in time
        GeoCircle(
            Coordinate(0.0, 1.0),
            500,
            dt=datetime(2020, 1, 1)
        ),
        # Intersects in timespan
        GeoCircle(
            Coordinate(0.0, 1.0),
            500,
            dt=TimeInterval(datetime(2019, 12, 1), datetime(2020, 1, 2))
        ),
        # Intersects in space, time eternal
        GeoPoint(
            Coordinate(0.0, 1.0),
            dt=None
        ),
    ])

def test_filter_by_property():
    collection = FeatureCollection(
        [
            GeoCircle(Coordinate(-0.118092, 51.509865), 500, properties={'example': 'red'}),
            GeoCircle(Coordinate(-0.141092, 51.529865), 500, properties={'example': 'property'}),
            GeoRing(Coordinate(-0.116092, 51.519865), inner_radius=800, outer_radius=1000, properties={'example': 'property'}),
            GeoRing(Coordinate(-0.101092, 51.514865), inner_radius=300, outer_radius=500, angle_min=60, angle_max=190, properties={'example': 'property1'}),
        ]
    )

    # Test for equality
    filtered = collection.filter_by_property('example', lambda x: x == 'red')
    assert len(filtered.geoshapes) == 1

    # Test for property not found
    with pytest.raises(KeyError):
        collection.filter_by_property('nonexistent_property', lambda x: x == 'red')

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


def test_collection_from_geojson():
    gjson = {
        'type': 'FeatureCollection',
        'features': [
            {
                'type': 'Feature',
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [[[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [0.0, 0.0]]]
                },
                'properties': {'datetime_start': '2020-01-01T00:00:00+00:00'},
                'id': 0
            },
            {
                'type': 'Feature',
                'geometry': {'type': 'Point', 'coordinates': [0.0, 2.0]},
                'properties': {
                    'datetime_start': '2020-01-01T00:00:00+00:00',
                    'datetime_end': '2020-01-02T00:00:00+00:00'
                },
                'id': 1
            },
            {
                'type': 'Feature',
                'geometry': {'type': 'LineString', 'coordinates': [[0.0, 0.0], [1.0, 1.0]]},
                'properties': {'datetime_end': '2020-01-02T00:00:00+00:00'},
                'id': 2
            },
            {
                'type': 'Feature',
                'geometry': {'type': 'MultiPoint', 'coordinates': [[0.0, 0.0], [1.0, 1.0]]},
                'properties': {},
                'id': 3
            },
            {
                'type': 'Feature',
                'geometry': {
                    'type': 'MultiLineString',
                    'coordinates': [
                        [[0.0, 0.0], [1.0, 1.0]],
                        [[1.0, 1.0], [2.0, 0.0]]
                    ]
                },
                'properties': {},
                'id': 3
            },
            {
                'type': 'Feature',
                'geometry': {
                    'type': 'MultiPolygon',
                    'coordinates': [
                        [[[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [0.0, 0.0]]],
                        [[[0.0, 0.0], [1.0, -1.0], [2.0, 0.0], [0.0, 0.0]]],
                    ]
                },
                'properties': {},
                'id': 3
            }
        ]
    }
    expected_shapes = [
        GeoPolygon([Coordinate(0.0, 0.0), Coordinate(1.0, 1.0), Coordinate(2.0, 0.0), Coordinate(0.0, 0.0)], dt=datetime(2020, 1, 1)),
        GeoPoint(Coordinate(0.0, 2.0), dt=TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 2))),
        GeoLineString([Coordinate(0.0, 0.0), Coordinate(1.0, 1.0)], dt=datetime(2020, 1, 2)),
        MultiGeoPoint([GeoPoint(Coordinate(0.0, 0.0)), GeoPoint(Coordinate(1.0, 1.0))]),
        MultiGeoLineString([
            GeoLineString([Coordinate(0.0, 0.0), Coordinate(1.0, 1.0)]),
            GeoLineString([Coordinate(1.0, 1.0), Coordinate(2.0, 0.0)]),
        ]),
        MultiGeoPolygon([
            GeoPolygon([Coordinate(0.0, 0.0), Coordinate(1.0, 1.0), Coordinate(2.0, 0.0), Coordinate(0.0, 0.0)]),
            GeoPolygon([Coordinate(0.0, 0.0), Coordinate(1.0, -1.0), Coordinate(2.0, 0.0), Coordinate(0.0, 0.0)]),
        ])
    ]
    expected = FeatureCollection(expected_shapes)
    assert FeatureCollection.from_geojson(gjson) == expected

    with pytest.raises(ValueError):
        gjson = {
            'type': 'Not a FeatureCollection',
            'features': []
        }
        _ = FeatureCollection.from_geojson(gjson)

    with pytest.raises(ValueError):
        gjson = {
            'type': 'FeatureCollection',
            'features': [
                {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Test',
                    }
                }
            ]
        }
        _ = FeatureCollection.from_geojson(gjson)


def test_collection_from_shapely():
    gls = GeoLineString([Coordinate(0.0, 0.0), Coordinate(1.0, 1.0), Coordinate(2.0, 2.0)])
    gpolygon = GeoPolygon([Coordinate(0.0, 0.0), Coordinate(1.0, 1.0), Coordinate(2.0, 0.0), Coordinate(0.0, 0.0)])
    gpoint = GeoPoint(Coordinate(0.0, 0.0))
    expected = FeatureCollection([gls, gpolygon, gpoint])

    gcol = shapely.GeometryCollection([x.to_shapely() for x in expected])
    assert FeatureCollection.from_shapely(gcol) == expected


def test_collection_geospan():
    col1 = FeatureCollection([
        GeoBox(Coordinate(-1.1, 0.), Coordinate(0., -5)),
        GeoBox(Coordinate(-0.5, 2.), Coordinate(2., -7)),
    ])
    assert col1.geospan == 12.1


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
    track = FeatureCollection(
        [
            GeoPoint(Coordinate(0.0, 0.0)),
            GeoLineString([Coordinate(0., 0.), Coordinate(1., 1.)]),
            GeoBox(Coordinate(0.5, 1.), Coordinate(1., 0.)),
            MultiGeoPoint([
                GeoPoint(Coordinate(0., 1.)),
                GeoPoint(Coordinate(0., 0.5))
            ]),
        ]
    )
    assert track.convex_hull == GeoPolygon([
        Coordinate(0., 0.), Coordinate(1., 0.),
        Coordinate(1., 1.), Coordinate(0., 1.,),
        Coordinate(0, 0.)
    ])


def test_collection_to_from_shapefile(caplog):
    # Tests both to and from because of temporary file usage
    # Shapes will be assigned an ID when written to shapefile, so have to hard code in tests

    # Test the various shapes get written/read correctly
    # When the file gets read all bounded shapes will become polygons - check this assumption in next test
    shapecol = FeatureCollection([
        GeoBox(Coordinate(0.0, 1.0), Coordinate(1.0, 0.0), properties={'ID': 0}).to_polygon(),
        GeoBox(Coordinate(0.0, 2.0), Coordinate(2.0, 0.0), properties={'ID': 1}).to_polygon(),
        GeoCircle(
            Coordinate(0.0, 2.0),
            1000,
            properties={'ID': 2},
            holes=[GeoCircle(Coordinate(0.0, 2.0), 500).to_polygon()]
        ).to_polygon(),
        GeoLineString([Coordinate(0.0, 1.0), Coordinate(1.0, 0.0)], properties={'ID': 0}),
        GeoLineString([Coordinate(0.0, 2.0), Coordinate(2.0, 0.0)], properties={'ID': 1}),
        GeoPoint(Coordinate(1.0, 0.0), properties={'ID': 0}),
        GeoPoint(Coordinate(2.0, 0.0), properties={'ID': 1}),
        MultiGeoPoint([GeoPoint(Coordinate(-1., 1.)), GeoPoint(Coordinate(-2, 1.))], properties={'ID': 0}),
        MultiGeoLineString([
            GeoLineString([Coordinate(0., 1.), Coordinate(1., 0.)]),
            GeoLineString([Coordinate(0., 0.), Coordinate(1., 1.)]),
        ], properties={'ID': 2}),
        MultiGeoPolygon(
            [
                GeoBox(
                    Coordinate(0., 1.),
                    Coordinate(1., 0.),
                    holes=[GeoBox(Coordinate(0.25, 0.75), Coordinate(0.75, 0.25)).to_polygon()]
                ).to_polygon(),
                GeoBox(Coordinate(1., 2.), Coordinate(2., 1.)).to_polygon(),
            ],
            properties={'ID': 2}
        )
    ])

    with tempfile.TemporaryDirectory() as f:
        with ZipFile(os.path.join(f, 'test.zip'), 'w') as zfile:
            shapecol.to_shapefile(zfile)

        new_shapecol = FeatureCollection.from_shapefile(os.path.join(f, 'test.zip'))
        assert set(new_shapecol.geoshapes) == set(shapecol.geoshapes)

    # Test again with Z and M values
    shapecol = FeatureCollection([
        GeoBox(
            Coordinate(0.0, 1.0, 99., 100.),
            Coordinate(1.0, 0.0, 99., 100.),
            properties={'ID': 0}
        ).to_polygon(),
        GeoLineString(
            [Coordinate(0.0, 1.0, 99., 100.), Coordinate(1.0, 0.0, 99., 100.)],
            properties={'ID': 0}
        ),
        GeoPoint(Coordinate(1.0, 0.0, 99., 100.), properties={'ID': 0}),
        MultiGeoPoint(
            [GeoPoint(Coordinate(-1., 1., 99., 100.)), GeoPoint(Coordinate(-2, 1., 99., 100.))],
            properties={'ID': 0}
        ),
        MultiGeoLineString([
            GeoLineString([Coordinate(0., 1., 99., 100.), Coordinate(1., 0., 99., 100.)]),
            GeoLineString([Coordinate(0., 0., 99., 100.), Coordinate(1., 1., 99., 100.)]),
        ], properties={'ID': 2}),
        MultiGeoPolygon(
            [
                GeoBox(
                    Coordinate(0., 1., 99., 100.),
                    Coordinate(1., 0., 99., 100.),
                    holes=[
                        GeoBox(Coordinate(0.25, 0.75, 99., 100.), Coordinate(0.75, 0.25, 99., 100.))
                    ]
                ).to_polygon(),
                GeoBox(Coordinate(1., 2., 99., 100.), Coordinate(2., 1., 99., 100.)).to_polygon(),
            ],
            properties={'ID': 2}
        )
    ])

    with tempfile.TemporaryDirectory() as f:
        with ZipFile(os.path.join(f, 'test.zip'), 'w') as zfile:
            shapecol.to_shapefile(zfile)

        new_shapecol = FeatureCollection.from_shapefile(os.path.join(f, 'test.zip'))
        for x in new_shapecol:
            if x not in shapecol:
                raise
        # assert set(new_shapecol.geoshapes) == set(shapecol.geoshapes)

    # Test that non-polygons get written/read correctly (should be read as a polygon)
    shapecol = FeatureCollection([
        GeoBox(Coordinate(0.0, 1.0), Coordinate(1.0, 0.0), properties={'ID': 0}),
        GeoCircle(Coordinate(0.0, 2.0), 1000, properties={'ID': 1}),
        GeoRing(Coordinate(0.0, 2.0), 1000, 500, properties={'ID': 2}),
    ])
    with tempfile.TemporaryDirectory() as f:
        with ZipFile(os.path.join(f, 'test.zip'), 'w') as zfile:
            shapecol.to_shapefile(zfile)

        new_shapecol = FeatureCollection.from_shapefile(os.path.join(f, 'test.zip'))
        assert set(new_shapecol.geoshapes) == set(x.to_polygon() for x in shapecol.geoshapes)

        new_shapecol = FeatureCollection.from_shapefile(os.path.join(f, 'test.zip'), read_layers=['nonexistent'])
        assert new_shapecol.geoshapes == []

    # Test writing/reading properties
    pointcol = FeatureCollection([
        GeoPoint(
            Coordinate(1.0, 0.0),
            dt=TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 2)),  # start and end date
            properties={
                'ID': 0,  # numeric
                'ex_prop': 'test2',  # string
                'ex2': True,  # boolean
            }
        ),
        GeoPoint(
            Coordinate(2.0, 0.0),
            dt=datetime(2020, 1, 1), # one date only
            properties={
                'ID': 1,   # numeric
                'ex_prop': 'test',  # string
                'ex2': False,  # boolean
            }
        ),
    ])
    with tempfile.TemporaryDirectory() as f:
        with ZipFile(os.path.join(f, 'test.zip'), 'w') as zfile:
            pointcol.to_shapefile(zfile)

        new_pointcol = FeatureCollection.from_shapefile(os.path.join(f, 'test.zip'))
        assert set(new_pointcol.geoshapes) == set(pointcol.geoshapes)

    # Test limiting the properties written
    with tempfile.TemporaryDirectory() as f:
        with ZipFile(os.path.join(f, 'test.zip'), 'w') as zfile:
            # should ignore the 'ex2' prop
            pointcol.to_shapefile(zfile, include_properties=['ex_prop'])

        new_pointcol = FeatureCollection.from_shapefile(os.path.join(f, 'test.zip'))

        expected = FeatureCollection([
            GeoPoint(
                Coordinate(1.0, 0.0),
                properties={'ID': 0, 'ex_prop': 'test2'}
            ),
            GeoPoint(
                Coordinate(2.0, 0.0),
                properties={'ID': 1, 'ex_prop': 'test'}
            ),
        ])
        assert set(new_pointcol.geoshapes) == set(expected.geoshapes)

    # Test that writing mixed property data types logs a warning
    with tempfile.TemporaryDirectory() as f:
        pointcol = FeatureCollection([
            GeoPoint(Coordinate(1.0, 0.0), properties={'ID': 0, 'prop': 1}),
            GeoPoint(Coordinate(2.0, 0.0), properties={'ID': 1, 'prop': '2'}),
        ])
        with ZipFile(os.path.join(f, 'test.zip'), 'w') as zfile:
            pointcol.to_shapefile(zfile)

        assert 'Conflicting data types found in properties; your shapefile may not get written correctly' in caplog.text


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
    assert track1.end == datetime(2020, 1, 3, tzinfo=timezone.utc)
    track1.geoshapes.pop(-1)
    assert track1.end == datetime(2020, 1, 2, tzinfo=timezone.utc)

    with pytest.raises(ValueError):
        _ = Track([]).end


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

    assert track1.time_start_diffs[0] == track1.geoshapes[1].dt.start - track1.geoshapes[0].dt.start
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


def test_track_filter_impossible_journeys(caplog):
    track = Track([
        GeoPoint(Coordinate(0., 0.), dt=datetime(2020, 1, 1)),
        GeoPoint(Coordinate(0.0001, 0.0001), dt=datetime(2020, 1, 1, 0, 1)),
        GeoPoint(Coordinate(1., 1.), dt=datetime(2020, 1, 1, 0, 2)),  # impossible
        GeoPoint(Coordinate(1., 1.), dt=datetime(2020, 1, 1, 0, 3)),  # impossible
        GeoPoint(Coordinate(0.0002, 0.0002), dt=datetime(2020, 1, 1, 0, 4)),
        GeoPoint(Coordinate(0.0002, 0.0002), dt=datetime(2020, 1, 1, 0, 5)),  # zero movement
        GeoPoint(Coordinate(0.0003, 0.0003), dt=datetime(2020, 1, 1, 0, 5)),  # zero timediff - removed without divide by zero
    ])
    assert track.filter_impossible_journeys(5) == Track([
        GeoPoint(Coordinate(0., 0.), dt=datetime(2020, 1, 1)),
        GeoPoint(Coordinate(0.0001, 0.0001), dt=datetime(2020, 1, 1, 0, 1)),
        GeoPoint(Coordinate(0.0002, 0.0002), dt=datetime(2020, 1, 1, 0, 4)),
        GeoPoint(Coordinate(0.0002, 0.0002), dt=datetime(2020, 1, 1, 0, 5))
    ])

    track = Track([
        GeoPoint(Coordinate(0., 0.), dt=datetime(2020, 1, 1)),
        GeoPoint(Coordinate(0.0001, 0.0001), dt=datetime(2020, 1, 1)),
    ])
    _ = track.filter_impossible_journeys(5)
    assert 'Duplicate timestamps detected' in caplog.text


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
    assert track1.filter_by_intersection(gbox) == track1

    gbox = GeoBox(Coordinate(0., 1.005), Coordinate(1., 0.))
    assert len(track1.filter_by_intersection(gbox)) == 6

    gbox = GeoBox(Coordinate(0., 2.), Coordinate(2., 0.), dt=datetime(2020, 1, 1, 3))
    assert len(track1.filter_by_intersection(gbox)) == 1

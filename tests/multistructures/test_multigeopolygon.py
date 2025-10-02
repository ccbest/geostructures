from datetime import datetime

import pytest
import shapely

from geostructures import MultiGeoPolygon, GeoBox, Coordinate, GeoCircle, GeoPolygon, FeatureCollection
from geostructures.time import TimeInterval

from tests.functions import pyshp_round_trip


def test_multigeopolygon_repr():
    mp = MultiGeoPolygon(
        [
            GeoBox(Coordinate(0., 1.), Coordinate(1., 0.)),
            GeoBox(Coordinate(1., 2.), Coordinate(2., 1.)),
        ]
    )
    assert repr(mp) == "<MultiGeoShape of 2 shapes>"

    mp = MultiGeoPolygon(
        [
            GeoBox(Coordinate(0., 1.), Coordinate(1., 0.)),
        ]
    )
    assert repr(mp) == "<MultiGeoShape of 1 shape>"


def test_multigeopolygon_area():
    box1 = GeoBox(Coordinate(0., 1.), Coordinate(1., 0.))
    box2 = GeoBox(Coordinate(1., 2.), Coordinate(2., 1.))
    mp = MultiGeoPolygon([box1, box2])
    assert mp.area == box1.area + box2.area

    mp = MultiGeoPolygon([box1])
    assert mp.area == box1.area

    mp = MultiGeoPolygon([])
    assert mp.area == 0.0


def test_multigeoshape_centroid():
    box1 = GeoBox(Coordinate(0., 1.), Coordinate(1., 0.))
    box2 = GeoBox(Coordinate(1., 2.), Coordinate(2., 1.))
    mp = MultiGeoPolygon([box1, box2])
    assert mp.centroid == Coordinate(1., 1.)


def test_multigeoshape_bounding_coords():
    box1 = GeoBox(Coordinate(0., 1.), Coordinate(1., 0.))
    box2 = GeoBox(Coordinate(1., 2.), Coordinate(2., 1.))
    mp = MultiGeoPolygon([box1, box2])
    assert mp.bounding_coords() == [
        GeoBox(Coordinate(0., 1.), Coordinate(1., 0.)).bounding_coords(),
        GeoBox(Coordinate(1., 2.), Coordinate(2., 1.)).bounding_coords()
    ]


def test_multigeoshape_bounding_edges():
    box1 = GeoBox(Coordinate(0., 1.), Coordinate(1., 0.))
    box2 = GeoBox(Coordinate(1., 2.), Coordinate(2., 1.))
    mp = MultiGeoPolygon([box1, box2])
    assert mp.bounding_edges() == [
        GeoBox(Coordinate(0., 1.), Coordinate(1., 0.)).bounding_edges(),
        GeoBox(Coordinate(1., 2.), Coordinate(2., 1.)).bounding_edges()
    ]


def test_multigeoshape_circumscribing_circle():
    box1 = GeoBox(Coordinate(0., 1.), Coordinate(1., 0.))
    box2 = GeoBox(Coordinate(1., 2.), Coordinate(2., 1.))
    mp = MultiGeoPolygon([box1, box2])
    assert mp.circumscribing_circle() == GeoCircle(Coordinate(1., 1.), 157249.38127194397)


def test_multigeoshape_copy():
    box1 = GeoBox(Coordinate(0., 1.), Coordinate(1., 0.))
    box2 = GeoBox(Coordinate(1., 2.), Coordinate(2., 1.))
    mp = MultiGeoPolygon([box1, box2])
    mp2 = mp.copy()
    assert mp2 == mp

    mp.geoshapes.pop()
    assert mp2 != mp


def test_multigeoshape_convex_hull():
    mp = MultiGeoPolygon([
        GeoBox(Coordinate(0., 1.), Coordinate(0.5, 0.)),
        GeoBox(Coordinate(0.5, 1.), Coordinate(1., 0.))
    ])
    assert mp.convex_hull() == GeoPolygon([
        Coordinate(0., 0.),
        Coordinate(1., 0.),
        Coordinate(1., 1.),
        Coordinate(0., 1.),
        Coordinate(0., 0.),
    ])


def test_multigeoshape_edges():
    mp = MultiGeoPolygon(
        [
            GeoBox(
                Coordinate(0., 1.),
                Coordinate(1., 0.),
                holes=[GeoBox(Coordinate(0.25, 0.75), Coordinate(0.75, 0.25))]
            ),
            GeoBox(Coordinate(1., 2.), Coordinate(2., 1.)),
        ]
    )
    assert mp.edges() == [
        GeoBox(
            Coordinate(0., 1.),
            Coordinate(1., 0.),
            holes=[GeoBox(Coordinate(0.25, 0.75), Coordinate(0.75, 0.25))]
        ).edges(),
        GeoBox(Coordinate(1., 2.), Coordinate(2., 1.)).edges(),
    ]


def test_multigeoshape_from_geojson():
    gjson = {
        "type": "Feature",
        "geometry": {
            "type": "MultiPolygon",
            "coordinates": [
                [
                    [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                    [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]
                ],
                [
                    [[1.0, 2.0], [1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]
                ]
            ]
        },
        "properties": {
            "test_prop": "test_val",
            "datetime_start": "2020-01-01T00:00:00+00:00",
            "datetime_end": "2020-01-01T00:00:00+00:00"
        },
        "test_kwarg": "test_kwarg"
    }
    mp = MultiGeoPolygon.from_geojson(gjson)
    assert mp.geoshapes == [
        GeoPolygon(
            [
                Coordinate(0.0, 1.0), Coordinate(0.0, 0.0),
                Coordinate(1.0, 0.0), Coordinate(1.0, 1.0),
                Coordinate(0.0, 1.0),
            ],
            holes=[
                GeoPolygon(
                    [
                        Coordinate(0.0, 1.0), Coordinate(0.0, 0.0),
                        Coordinate(1.0, 0.0), Coordinate(1.0, 1.0),
                        Coordinate(0.0, 1.0),
                    ],
                )
            ]
        ),
        GeoPolygon(
            [
                Coordinate(1.0, 2.0), Coordinate(1.0, 1.0),
                Coordinate(2.0, 1.0), Coordinate(2.0, 2.0),
                Coordinate(1.0, 2.0),
            ],
        )
    ]
    assert mp._properties == {'test_prop': 'test_val'}
    assert mp.dt == TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 1))

    # Only geo interface
    gjson = gjson['geometry']
    mp = MultiGeoPolygon.from_geojson(gjson)
    assert mp.geoshapes == [
        GeoPolygon(
            [
                Coordinate(0.0, 1.0), Coordinate(0.0, 0.0),
                Coordinate(1.0, 0.0), Coordinate(1.0, 1.0),
                Coordinate(0.0, 1.0),
            ],
            holes=[
                GeoPolygon(
                    [
                        Coordinate(0.0, 1.0), Coordinate(0.0, 0.0),
                        Coordinate(1.0, 0.0), Coordinate(1.0, 1.0),
                        Coordinate(0.0, 1.0),
                    ],
                )
            ]
        ),
        GeoPolygon(
            [
                Coordinate(1.0, 2.0), Coordinate(1.0, 1.0),
                Coordinate(2.0, 1.0), Coordinate(2.0, 2.0),
                Coordinate(1.0, 2.0),
            ],
        )
    ]


    with pytest.raises(ValueError):
        gjson = {
            "type": "Feature",
            "geometry": {
                "type": "test",
            }
        }
        MultiGeoPolygon.from_geojson(gjson)


def test_multigeoshape_from_shapely():
    shapely_mp = shapely.MultiPolygon([
        [
            [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            [[[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]]
        ],
        [
            [[1.0, 2.0], [1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]
        ]
    ])
    mp = MultiGeoPolygon.from_shapely(shapely_mp)
    assert mp.geoshapes == [
        GeoPolygon(
            [
                Coordinate(0.0, 1.0), Coordinate(0.0, 0.0),
                Coordinate(1.0, 0.0), Coordinate(1.0, 1.0),
                Coordinate(0.0, 1.0),
            ],
            holes=[
                GeoPolygon(
                    [
                        Coordinate(0.0, 1.0), Coordinate(0.0, 0.0),
                        Coordinate(1.0, 0.0), Coordinate(1.0, 1.0),
                        Coordinate(0.0, 1.0),
                    ],
                )
            ]
        ),
        GeoPolygon(
            [
                Coordinate(1.0, 2.0), Coordinate(1.0, 1.0),
                Coordinate(2.0, 1.0), Coordinate(2.0, 2.0),
                Coordinate(1.0, 2.0),
            ],
        )
    ]


def test_multigeoshape_from_wkt():
    wkt = "MULTIPOLYGON (((0 1, 0 0, 1 0, 1 1, 0 1), (0 1, 1 1, 1 0, 0 0, 0 1)), ((1 2, 1 1, 2 1, 2 2, 1 2)))"
    mp = MultiGeoPolygon.from_wkt(wkt)
    assert mp.geoshapes == [
        GeoPolygon(
            [
                Coordinate(0.0, 1.0), Coordinate(0.0, 0.0),
                Coordinate(1.0, 0.0), Coordinate(1.0, 1.0),
                Coordinate(0.0, 1.0),
            ],
            holes=[
                GeoPolygon(
                    [
                        Coordinate(0.0, 1.0), Coordinate(0.0, 0.0),
                        Coordinate(1.0, 0.0), Coordinate(1.0, 1.0),
                        Coordinate(0.0, 1.0),
                    ],
                )
            ]
        ),
        GeoPolygon(
            [
                Coordinate(1.0, 2.0), Coordinate(1.0, 1.0),
                Coordinate(2.0, 1.0), Coordinate(2.0, 2.0),
                Coordinate(1.0, 2.0),
            ],
        )
    ]

    with pytest.raises(ValueError):
        MultiGeoPolygon.from_wkt('test')


def test_multigeoshape_linear_rings():
    mp = MultiGeoPolygon(
        [
            GeoBox(
                Coordinate(0., 1.),
                Coordinate(1., 0.),
                holes=[GeoBox(Coordinate(0.25, 0.75), Coordinate(0.75, 0.25))]
            ),
            GeoBox(Coordinate(1., 2.), Coordinate(2., 1.)),
        ]
    )
    assert mp.linear_rings() == [
        GeoBox(
            Coordinate(0., 1.),
            Coordinate(1., 0.),
            holes=[GeoBox(Coordinate(0.25, 0.75), Coordinate(0.75, 0.25))]
        ).linear_rings(),
        GeoBox(Coordinate(1., 2.), Coordinate(2., 1.)).linear_rings()
    ]


def test_multigeoshape_to_geojson():
    mp = MultiGeoPolygon(
        [
            GeoBox(
                Coordinate(0., 1.),
                Coordinate(1., 0.),
                holes=[GeoBox(Coordinate(0.25, 0.75), Coordinate(0.75, 0.25))]
            ),
            GeoBox(Coordinate(1., 2.), Coordinate(2., 1.)),
        ],
        dt=TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 1)),
        properties={'test_prop': 'test_val'}
    )
    assert mp.to_geojson(test_kwarg="test_kwarg") == {
        "type": "Feature",
        "geometry": {
            "type": "MultiPolygon",
            "coordinates": [
                [
                    [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                    [[0.25, 0.75], [0.75, 0.75], [0.75, 0.25], [0.25, 0.25], [0.25, 0.75]]
                ],
                [
                    [[1.0, 2.0], [1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]
                ]
            ]
        },
        "properties": {
            "test_prop": "test_val",
            "datetime_start": "2020-01-01T00:00:00+00:00",
            "datetime_end": "2020-01-01T00:00:00+00:00"
        },
        "test_kwarg": "test_kwarg"
    }


def test_multigeopolygon_to_pyshp(pyshp_round_trip):

    original = FeatureCollection([
            MultiGeoPolygon(
            [
                GeoBox(Coordinate(0., 1., z=1., m=3.), Coordinate(1., 0., z=1., m=3.)).to_polygon(),
                GeoBox(Coordinate(1., 2., z=1., m=3.), Coordinate(2., 1., z=1., m=3.)).to_polygon(),
            ],
            dt=datetime(2020, 1, 1),
            properties={'test': 'prop'}
        )
    ])
    new = pyshp_round_trip(original)
    assert set(original.geoshapes) == set(new.geoshapes)

    original = FeatureCollection([
            MultiGeoPolygon(
            [
                GeoBox(Coordinate(0., 1., m=3.), Coordinate(1., 0., m=3.)).to_polygon(),
                GeoBox(Coordinate(1., 2., m=3.), Coordinate(2., 1., m=3.)).to_polygon(),
            ],
            dt=datetime(2020, 1, 1),
            properties={'test': 'prop'}
        )
    ])
    new = pyshp_round_trip(original)
    assert set(original.geoshapes) == set(new.geoshapes)


def test_multigeoshape_to_shapely():
    mp = MultiGeoPolygon(
        [
            GeoBox(
                Coordinate(0., 1.),
                Coordinate(1., 0.),
                holes=[GeoBox(Coordinate(0.25, 0.75), Coordinate(0.75, 0.25))]
            ),
            GeoBox(Coordinate(1., 2.), Coordinate(2., 1.)),
        ],
    )
    assert mp.to_shapely() == shapely.MultiPolygon([
        [
            [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            [[[0.25, 0.75], [0.75, 0.75], [0.75, 0.25], [0.25, 0.25], [0.25, 0.75]]]
        ],
        [
            [[1.0, 2.0], [1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]
        ]
    ])


def test_multigeoshape_to_wkt():
    mp = MultiGeoPolygon(
        [
            GeoBox(
                Coordinate(0., 1.),
                Coordinate(1., 0.),
                holes=[GeoBox(Coordinate(1.0, 0.0), Coordinate(0.0, 1.0))]
            ),
            GeoBox(Coordinate(1., 2.), Coordinate(2., 1.)),
        ]
    )
    assert mp.to_wkt() == (
        "MULTIPOLYGON("
        "((0.0 1.0,0.0 0.0,1.0 0.0,1.0 1.0,0.0 1.0), (1.0 0.0,0.0 0.0,0.0 1.0,1.0 1.0,1.0 0.0)), "
        "((1.0 2.0,1.0 1.0,2.0 1.0,2.0 2.0,1.0 2.0))"
        ")"
    )

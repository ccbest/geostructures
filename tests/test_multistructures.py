
from datetime import datetime

import pytest
import shapely

from geostructures import *
from geostructures.multistructures import *
from geostructures.time import TimeInterval


def test_multigeolinestring_contains():
    mls = MultiGeoLineString(
        [
            GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5), Coordinate(1., 0.)]),
            GeoLineString([Coordinate(1., 1.), Coordinate(0.5, 0.5), Coordinate(0., 0.)]),
        ],
        dt=TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 2))
    )
    assert Coordinate(0.5, 0.5) in mls
    assert GeoPoint(Coordinate(0.5, 0.5)) in mls
    assert GeoPoint(Coordinate(0.5, 0.5), dt=datetime(2020, 1, 1, 1))


def test_multigeolinestring_hash():
    mls_1 = MultiGeoLineString(
        [
            GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5), Coordinate(1., 0.)]),
            GeoLineString([Coordinate(1., 1.), Coordinate(0.5, 0.5), Coordinate(0., 0.)]),
        ],
        dt=TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 2))
    )
    mls_2 = MultiGeoLineString(
        [
            GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5), Coordinate(1., 0.)]),
            GeoLineString([Coordinate(1., 1.), Coordinate(0.5, 0.5), Coordinate(0., 0.)]),
        ],
        dt=TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 2))
    )
    assert hash(mls_1) == hash(mls_2)

    mls_2 = MultiGeoLineString(
        [
            GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5), Coordinate(1., 0.)]),
            GeoLineString([Coordinate(1., 1.), Coordinate(0.5, 0.5), Coordinate(0., 0.)]),
        ],
        dt=TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 3))
    )
    assert hash(mls_1) != hash(mls_2)


def test_multigeolinestring_repr():
    mls = MultiGeoLineString(
        [
            GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5), Coordinate(1., 0.)]),
        ],
    )
    assert repr(mls) == '<MultiGeoLineString of 1 linestring>'

    mls = MultiGeoLineString(
        [
            GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5), Coordinate(1., 0.)]),
            GeoLineString([Coordinate(1., 1.), Coordinate(0.5, 0.5), Coordinate(0., 0.)]),
        ],
    )
    assert repr(mls) == '<MultiGeoLineString of 2 linestrings>'


def test_multigeolinestring_bounds():
    mls = MultiGeoLineString(
        [
            GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5), Coordinate(1., 0.)]),
            GeoLineString([Coordinate(2., 1.), Coordinate(0.5, 0.5), Coordinate(0., 0.)]),
        ],
    )
    assert mls.bounds() == ((0., 2.), (0., 1.))


def test_multigeolinestring_circumscribing_circle():
    mls = MultiGeoLineString(
        [
            GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5), Coordinate(1., 0.)]),
            GeoLineString([Coordinate(1., 1.), Coordinate(0.5, 0.5), Coordinate(0., 0.)]),
        ],
    )
    assert mls.circumscribing_circle() == GeoCircle(Coordinate(0.5, 0.5), 78626.18767687456)


def test_multigeolinestring_contains_coordinate():
    mls = MultiGeoLineString(
        [
            GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5), Coordinate(1., 0.)]),
            GeoLineString([Coordinate(1., 1.), Coordinate(0.5, 0.5), Coordinate(0., 0.)]),
        ],
    )
    assert mls.contains_coordinate(Coordinate(0.5, 0.5))
    assert not mls.contains_coordinate(Coordinate(2.0, 2.0))


def test_multigeolinestring_contains_shape():
    mls = MultiGeoLineString(
        [
            GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5), Coordinate(1., 0.)]),
            GeoLineString([Coordinate(1., 1.), Coordinate(0.5, 0.5), Coordinate(0., 0.)]),
        ],
    )
    assert mls.contains_shape(
        MultiGeoLineString(
            [
                GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5)]),
                GeoLineString([Coordinate(1., 1.), Coordinate(0.5, 0.5)]),
            ],
        )
    )
    assert mls.contains_shape(
        MultiGeoPoint([GeoPoint(Coordinate(0., 1.)), GeoPoint(Coordinate(1., 1.))])
    )

    assert not mls.contains_shape(GeoCircle(Coordinate(0., 0.), 500))
    assert not mls.contains_shape(GeoPoint(Coordinate(2., 2.)))


def test_multigeolinestring_copy():
    mls_1 = MultiGeoLineString(
        [
            GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5), Coordinate(1., 0.)]),
            GeoLineString([Coordinate(1., 1.), Coordinate(0.5, 0.5), Coordinate(0., 0.)]),
        ],
    )
    mls_2 = mls_1.copy()
    assert mls_1 == mls_2

    mls_2.geoshapes.pop(0)
    assert mls_1 != mls_2


def test_multigeolinestring_from_geojson():
    gjson = {
        "type": "Feature",
        "geometry": {
            "type": "MultiLineString",
            "coordinates": [
                [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]],
                [[1.0, 1.0], [0.5, 0.5], [0.0, 0.0]]
            ]
        },
        "properties": {
            "test_prop": "test_value",
            "datetime_start": "2020-01-01 00:00:00.000",
            "datetime_end": "2020-01-02 00:00:00.000",
        }
    }
    mls = MultiGeoLineString.from_geojson(gjson)
    assert mls.geoshapes == [
        GeoLineString([Coordinate(0.0, 1.0), Coordinate(0.5, 0.5), Coordinate(1.0, 0.0)]),
        GeoLineString([Coordinate(1.0, 1.0), Coordinate(0.5, 0.5), Coordinate(0.0, 0.0)])
    ]
    assert mls.dt == TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 2))
    assert mls._properties == {'test_prop': 'test_value'}


def test_multigeolinestring_from_shapely():
    smls = shapely.MultiLineString([[[0, 0], [1, 2]], [[4, 4], [5, 6]]])
    mls = MultiGeoLineString.from_shapely(smls)
    expected = [
        GeoLineString([Coordinate(0.0, 0.0), Coordinate(1., 2.)]),
        GeoLineString([Coordinate(4.0, 4.0), Coordinate(5., 6.)])
    ]
    assert all([a == b for a, b in zip(mls.geoshapes, expected)])


def test_multigeolinestring_from_wkt():
    wkt = "MULTILINESTRING((0.0 1.0,0.5 0.5,1.0 0.0), (1.0 1.0,0.5 0.5,0.0 0.0))"
    mls = MultiGeoLineString.from_wkt(wkt)
    assert mls.geoshapes == [
        GeoLineString([Coordinate(0.0, 1.0), Coordinate(0.5, 0.5), Coordinate(1.0, 0.0)]),
        GeoLineString([Coordinate(1.0, 1.0), Coordinate(0.5, 0.5), Coordinate(0.0, 0.0)])
    ]


def test_multigeolinestring_intersects_shape():
    mls = MultiGeoLineString(
        [
            GeoLineString([Coordinate(0., 1.), Coordinate(0.5, 0.5), Coordinate(1., 0.)]),
            GeoLineString([Coordinate(1., 1.), Coordinate(0.5, 0.5), Coordinate(0., 0.)]),
        ],
    )
    assert mls.intersects_shape(
        MultiGeoLineString(
            [
                GeoLineString([Coordinate(0., 0.5), Coordinate(1.0, 0.5)]),
            ],
        )
    )

    assert mls.intersects_shape(GeoLineString([Coordinate(0., 0.5), Coordinate(1.0, 0.5)]))


def test_multigeolinestring_to_geojson():
    mls = MultiGeoLineString(
        [
            GeoLineString([Coordinate(0.0, 1.0), Coordinate(0.5, 0.5), Coordinate(1.0, 0.0)]),
            GeoLineString([Coordinate(1.0, 1.0), Coordinate(0.5, 0.5), Coordinate(0.0, 0.0)])
        ],
        dt=datetime(2020, 1, 1),
        properties={'test_prop': 'test_value'}
    )
    assert mls.to_geojson(test_kwarg='test_kwarg') == {
        "type": "Feature",
        "geometry": {
            "type": "MultiLineString",
            "coordinates": [
                [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]],
                [[1.0, 1.0], [0.5, 0.5], [0.0, 0.0]]
            ]
        },
        "properties": {
            "test_prop": "test_value",
            "datetime_start": "2020-01-01T00:00:00+00:00",
            "datetime_end": "2020-01-01T00:00:00+00:00",
        },
        'test_kwarg': 'test_kwarg'
    }


def test_multigeolinestring_to_shapely():
    smls = shapely.MultiLineString([[[0, 0], [1, 2]], [[4, 4], [5, 6]]])
    mls = MultiGeoLineString([
        GeoLineString([Coordinate(0.0, 0.0), Coordinate(1., 2.)]),
        GeoLineString([Coordinate(4.0, 4.0), Coordinate(5., 6.)])
    ])
    assert mls.to_shapely() == smls


def test_multigeolinestring_to_wkt():
    mls = MultiGeoLineString([
        GeoLineString([Coordinate(0.0, 1.0), Coordinate(0.5, 0.5), Coordinate(1.0, 0.0)]),
        GeoLineString([Coordinate(1.0, 1.0), Coordinate(0.5, 0.5), Coordinate(0.0, 0.0)])
    ])
    assert mls.to_wkt() == "MULTILINESTRING((0.0 1.0,0.5 0.5,1.0 0.0), (1.0 1.0,0.5 0.5,0.0 0.0))"


def test_multigeopoint_repr():
    mp = MultiGeoPoint([
        GeoPoint(Coordinate(0., 0.)),
        GeoPoint(Coordinate(1., 1.))
    ])
    assert repr(mp) == '<MultiGeoPoint of 2 points>'

    mp = MultiGeoPoint([
        GeoPoint(Coordinate(0., 0.)),
    ])
    assert repr(mp) == '<MultiGeoPoint of 1 point>'


def test_multigeopoint_area():
    mp = MultiGeoPoint([
        GeoPoint(Coordinate(0., 0.)),
        GeoPoint(Coordinate(1., 1.))
    ])
    assert mp.area() == 0.


def test_multigeopoint_bounds():
    mp = MultiGeoPoint([
        GeoPoint(Coordinate(0., 0.)),
        GeoPoint(Coordinate(1., 1.))
    ])
    assert mp.bounds() == ((0., 1.), (0., 1.))


def test_multigeopoint_centroid():
    mp = MultiGeoPoint([
        GeoPoint(Coordinate(0., 0.)),
        GeoPoint(Coordinate(1., 1.))
    ])
    assert mp.centroid == Coordinate(0.5, 0.5)


def test_multigeopoint_circumscribing_circle():
    mp = MultiGeoPoint([
        GeoPoint(Coordinate(0., 0.)),
        GeoPoint(Coordinate(1., 1.))
    ])
    assert mp.circumscribing_circle() == GeoCircle(Coordinate(0.5, 0.5), 78626.18767687456)


def test_multigeopoint_from_geojson():
    gjson = {
        "type": "Feature",
        "geometry": {
            "type": "MultiPoint",
            "coordinates": [[0.0, 1.0], [1.0, 1.0]]},
        "properties": {
            "test_prop": "test_value",
            "datetime_start": "2020-01-01T00:00:00+00:00",
            "datetime_end": "2020-01-01T00:00:00+00:00"
        }
    }
    mp = MultiGeoPoint.from_geojson(gjson)
    assert mp.geoshapes == [
        GeoPoint(Coordinate(0., 1.)),
        GeoPoint(Coordinate(1., 1.))
    ]
    assert mp._properties == {"test_prop": "test_value"}
    assert mp.dt == TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 1))


def test_multigeopoint_from_shapely():
    shapely_mp = shapely.MultiPoint([(0., 1.), (1., 1.)])
    mp = MultiGeoPoint.from_shapely(shapely_mp)
    assert mp.geoshapes == [
        GeoPoint(Coordinate(0., 1.)),
        GeoPoint(Coordinate(1., 1.))
    ]


def test_multigeopoint_from_wkt():
    wkt = "MULTIPOINT (0 1, 1 1)"
    mp = MultiGeoPoint.from_wkt(wkt)
    assert mp.geoshapes == [
        GeoPoint(Coordinate(0., 1.)),
        GeoPoint(Coordinate(1., 1.))
    ]


def test_multigeopoint_to_geojson():
    mp = MultiGeoPoint(
        [
            GeoPoint(Coordinate(0., 1.)),
            GeoPoint(Coordinate(1., 1.))
        ],
        dt=datetime(2020, 1, 1),
        properties={'test_prop': 'test_val'},
    )
    assert mp.to_geojson(test_kwarg="test_kwarg") == {
        "type": "Feature",
        "geometry": {
            "type": "MultiPoint",
            "coordinates": [[0.0, 1.0], [1.0, 1.0]]},
        "properties": {
            "test_prop": "test_val",
            "datetime_start": "2020-01-01T00:00:00+00:00",
            "datetime_end": "2020-01-01T00:00:00+00:00"
        },
        "test_kwarg": "test_kwarg"
    }


def test_multigeopoint_to_shapely():
    mp = MultiGeoPoint(
        [
            GeoPoint(Coordinate(0., 1.)),
            GeoPoint(Coordinate(1., 1.))
        ],
    )
    assert mp.to_shapely() == shapely.MultiPoint([(0., 1.), (1., 1.)])


def test_multigeopoint_to_wkt():
    mp = MultiGeoPoint([
        GeoPoint(Coordinate(0., 1.)),
        GeoPoint(Coordinate(1., 1.))
    ])
    assert mp.to_wkt() == "MULTIPOINT(0.0 1.0, 1.0 1.0)"


def test_multigeoshape_repr():
    mp = MultiGeoShape(
        [
            GeoBox(Coordinate(0., 1.), Coordinate(1., 0.)),
            GeoBox(Coordinate(1., 2.), Coordinate(2., 1.)),
        ]
    )
    assert repr(mp) == "<MultiGeoShape of 2 shapes>"

    mp = MultiGeoShape(
        [
            GeoBox(Coordinate(0., 1.), Coordinate(1., 0.)),
        ]
    )
    assert repr(mp) == "<MultiGeoShape of 1 shape>"


def test_multigeoshape_area():
    mp = MultiGeoShape(
        [
            GeoBox(Coordinate(0., 1.), Coordinate(1., 0.)),
            GeoBox(Coordinate(1., 2.), Coordinate(2., 1.)),
        ]
    )
    assert mp.area == 12305128751.042904 + 12308778361.469452


def test_multigeoshape_bounds():
    mp = MultiGeoShape(
        [
            GeoBox(Coordinate(0., 1.), Coordinate(1., 0.)),
            GeoBox(Coordinate(1., 2.), Coordinate(2., 1.)),
        ]
    )
    assert mp.bounds == ((0., 2.), (0., 2.))


def test_multigeoshape_centroid():
    mp = MultiGeoShape(
        [
            GeoBox(Coordinate(0., 1.), Coordinate(1., 0.)),
            GeoBox(Coordinate(1., 2.), Coordinate(2., 1.)),
        ]
    )
    assert mp.centroid == Coordinate(1., 1.)


def test_multigeoshape_bounding_coords():
    mp = MultiGeoShape(
        [
            GeoBox(Coordinate(0., 1.), Coordinate(1., 0.)),
            GeoBox(Coordinate(1., 2.), Coordinate(2., 1.)),
        ]
    )
    assert mp.bounding_coords() == [
        GeoBox(Coordinate(0., 1.), Coordinate(1., 0.)).bounding_coords(),
        GeoBox(Coordinate(1., 2.), Coordinate(2., 1.)).bounding_coords()
    ]


def test_multigeoshape_bounding_edges():
    mp = MultiGeoShape(
        [
            GeoBox(Coordinate(0., 1.), Coordinate(1., 0.)),
            GeoBox(Coordinate(1., 2.), Coordinate(2., 1.)),
        ]
    )
    assert mp.bounding_edges() == [
        GeoBox(Coordinate(0., 1.), Coordinate(1., 0.)).bounding_edges(),
        GeoBox(Coordinate(1., 2.), Coordinate(2., 1.)).bounding_edges()
    ]


def test_multigeoshape_circumscribing_circle():
    mp = MultiGeoShape(
        [
            GeoBox(Coordinate(0., 1.), Coordinate(1., 0.)),
            GeoBox(Coordinate(1., 2.), Coordinate(2., 1.)),
        ]
    )
    assert mp.circumscribing_circle() == GeoCircle(Coordinate(1., 1.), 157249.38127194397)


def test_multigeoshape_edges():
    mp = MultiGeoShape(
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
    mp = MultiGeoShape.from_geojson(gjson)
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
    mp = MultiGeoShape.from_shapely(shapely_mp)
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
    mp = MultiGeoShape.from_wkt(wkt)
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

def test_multigeoshape_linear_rings():
    mp = MultiGeoShape(
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
    mp = MultiGeoShape(
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


def test_multigeoshape_to_shapely():
    mp = MultiGeoShape(
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
    mp = MultiGeoShape(
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

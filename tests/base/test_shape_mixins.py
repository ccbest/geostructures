
from datetime import datetime, timezone

from fastkml import Placemark
from fastkml.times import TimeStamp, KmlDateTime
from fastkml.data import ExtendedData, Data
from pygeoif import MultiPolygon
import pytest

from geostructures import (
    Coordinate, GeoBox, GeoCircle, GeoPoint, GeoPolygon, MultiGeoPoint, MultiGeoPolygon,
)
from geostructures.geodesic import destination_point
from geostructures.time import TimeInterval

from tests.functions import default_test_datetime


def test_polygonbase_holes_cannot_have_holes():
    with pytest.raises(ValueError):
        _ = GeoCircle(
            Coordinate(0.0, 0.0), 1000,
            # Hole shape itself has a hole
            holes=[GeoCircle(Coordinate(0.0, 0.0), 500, holes=[GeoCircle(Coordinate(0.0, 0.0), 250)])]
        )


def test_polygonbase_area():
    assert GeoBox(Coordinate(0.0, 1.0), Coordinate(1., 0.)).area == 12308778361.469452


def test_polygonbase_volume():
    assert GeoBox(
        Coordinate(0.0, 1.0),
        Coordinate(1., 0.),
        dt=TimeInterval(datetime(2020, 1, 1, 1, 1, 1), datetime(2020, 1, 1, 1, 1, 2))
    ).volume == 12308778361.469452

    assert GeoBox(
        Coordinate(0.0, 1.0),
        Coordinate(1., 0.),
        dt=TimeInterval(datetime(2020, 1, 1, 1, 1, 1), datetime(2020, 1, 1, 1, 1, 3))
    ).volume == 24617556722.938904

    assert GeoBox(Coordinate(0.0, 1.0), Coordinate(1., 0.)).volume == 0.
    assert GeoBox(Coordinate(0.0, 1.0), Coordinate(1., 0.), dt=datetime(2020, 1, 1)).volume == 0.


def test_polygonbase_has_m():
    geo = GeoCircle(Coordinate('0.0', '0.0'), 500, dt=default_test_datetime)
    assert not geo.has_m

    geo = GeoPolygon([
        Coordinate('0.0', '0.0', m=1),
        Coordinate('1.0', '1.0', m=1),
        Coordinate('1.0', '0.0', m=1),
        Coordinate('0.0', '0.0', m=1),
    ])
    assert geo.has_m


def test_polygonbase_has_z():
    geo = GeoCircle(Coordinate('0.0', '0.0'), 500)
    assert not geo.has_z

    geo = GeoCircle(Coordinate('0.0', '0.0', z=1), 500)
    assert geo.has_z


def test_polygonbase_bounding_edges():
    poly = GeoPolygon([Coordinate(1.0, 0.0), Coordinate(1.0, 1.0), Coordinate(0.0, 0.5), Coordinate(1.0, 0.0)])
    assert poly.bounding_edges() == [
        (Coordinate(1.0, 0.0), Coordinate(1.0, 1.0)),
        (Coordinate(1.0, 1.0), Coordinate(0.0, 0.5)),
        (Coordinate(0.0, 0.5), Coordinate(1.0, 0.0)),
        (Coordinate(1.0, 0.0), Coordinate(1.0, 0.0))
    ]


def test_polygonbase_contains_coordinate():
    # Triangle
    polygon = GeoPolygon([
        Coordinate(0.0, 0.0), Coordinate(1.0, 1.0),
        Coordinate(1.0, 0.0), Coordinate(0.0, 0.0)
    ])
    # Way outside
    assert Coordinate(1.5, 1.5) not in polygon

    # Center along hypotenuse - boundary intersection should not count
    assert Coordinate(0.5, 0.5) not in polygon

    # Nudge above to be just inside
    assert Coordinate(0.5, 0.49) in polygon

    # Outside, to upper left
    assert Coordinate(0.1, 0.9) not in polygon

    # 5-point Star
    polygon = GeoPolygon([
        Coordinate(0.004, 0.382), Coordinate(0.596, 0.803), Coordinate(0.364, 0.114),
        Coordinate(0.948, -0.319), Coordinate(0.221, -0.311), Coordinate(-0.01, -1),
        Coordinate(-0.228, -0.307), Coordinate(-0.954, -0.299), Coordinate(-0.362, 0.122),
        Coordinate(-0.579, 0.815), Coordinate(0.004, 0.382)
    ])
    assert Coordinate(0.0, 0.0) in polygon
    assert Coordinate(0.9, 0.1) not in polygon
    assert Coordinate(-0.9, 0.4) not in polygon
    assert Coordinate(-0.9, 0.1) not in polygon

    # Box with hole in middle
    polygon = GeoPolygon(
        [
            Coordinate(0.0, 0.0), Coordinate(0.0, 1.0), Coordinate(1.0, 1.0),
            Coordinate(1.0, 0.0), Coordinate(0.0, 0.0)
        ],
        holes=[GeoPolygon([
            Coordinate(0.25, 0.25), Coordinate(0.25, 0.75), Coordinate(0.75, 0.75),
            Coordinate(0.75, 0.25), Coordinate(0.25, 0.25)
        ])]
    )
    assert Coordinate(0.9, 0.9) in polygon  # outside hole
    assert Coordinate(0.5, 0.5) not in polygon  # inside hole
    assert Coordinate(0.75, 0.75) in polygon  # on hole edge


def test_polygonbase_contains_shape():
    # Base case
    circle_outer = GeoCircle(Coordinate(0., 0.), 5_000)
    circle_inner = GeoCircle(Coordinate(0., 0.), 2_000)
    assert circle_outer.contains_shape(circle_inner)
    assert not circle_inner.contains_shape(circle_outer)

    # Intersecting, not containing
    circle1 = GeoCircle(Coordinate(0., 0.), 5_000)
    circle2 = GeoCircle(Coordinate(0.0899322, 0.0), 6_000)
    assert not circle1.contains_shape(circle2)

    # inner circle fully contained within hole
    circle_outer = GeoCircle(Coordinate(0., 0.,), 5_000, holes=[GeoCircle(Coordinate(0., 0.,), 4_000)])
    circle_inner = GeoCircle(Coordinate(0., 0.), 2_000)
    assert not circle_outer.contains_shape(circle_inner)

    # Verify it works for multishapes
    mp = MultiGeoPoint([GeoPoint(Coordinate(0., 0.)), GeoPoint(Coordinate(0.00001, 0.00001))])
    assert circle1.contains(mp)

    mp = MultiGeoPoint([GeoPoint(Coordinate(0., 0.)), GeoPoint(Coordinate(1.0, 1.0))])
    assert not circle1.contains(mp)


def test_polygonbase_intersects_shape():
    circle1 = GeoCircle(Coordinate(0.0, 0.0), 5_000)

    new_centroid = destination_point(circle1.centroid, 90, 10_000)  # Exactly 10km to the right
    circle2 = GeoCircle(new_centroid, 5_000)

    # Exactly one point where shapes intersect (boundary)
    assert circle1.intersects_shape(circle2)
    assert circle2.intersects_shape(circle1)

    new_centroid = destination_point(circle1.centroid, 90, 9_999)  # Nudged just barely to the left
    circle2 = GeoCircle(new_centroid, 5_000)
    assert circle1.intersects_shape(circle2)
    assert circle2.intersects_shape(circle1)

    new_centroid = destination_point(circle1.centroid, 90, 10_001)  # Nudged just barely to the right
    circle2 = GeoCircle(new_centroid, 5_000)
    assert not circle1.intersects_shape(circle2)
    assert not circle2.intersects_shape(circle1)

    circle1 = GeoCircle(Coordinate(0.0, 0.0), 5_000)
    circle2 = GeoCircle(Coordinate(0.0, 0.0), 2_000)  # Fully contained
    assert circle1.intersects_shape(circle2)
    assert circle2.intersects_shape(circle1)

    # points
    point = GeoPoint(Coordinate(0., 0.))
    assert circle1.intersects_shape(point)
    assert point.intersects_shape(circle1)

    # multishapes
    assert circle1.intersects_shape(
        MultiGeoPoint([
            GeoPoint(Coordinate(0., 0.)),
            GeoPoint(Coordinate(0.001, 0.001))
        ])
    )
    assert not circle1.intersects_shape(
        MultiGeoPoint([
            GeoPoint(Coordinate(1., 0.)),
            GeoPoint(Coordinate(1., 0.001))
        ])
    )


def test_polygonbase_edges():
    polygon = GeoPolygon(
        [
            Coordinate(1.0, 0.0), Coordinate(1.0, 1.0),
            Coordinate(0.0, 0.5), Coordinate(1.0, 0.0)
        ],
        holes=[
            GeoPolygon([
                Coordinate(0.4, 0.6), Coordinate(0.5, 0.4),
                Coordinate(0.6, 0.6), Coordinate(0.4, 0.6),
            ])
        ]
    )
    assert polygon.edges() == [
        [
            (Coordinate(1.0, 0.0), Coordinate(1.0, 1.0)),
            (Coordinate(1.0, 1.0), Coordinate(0.0, 0.5)),
            (Coordinate(0.0, 0.5), Coordinate(1.0, 0.0)),
        ],
        [
            (Coordinate(0.4, 0.6), Coordinate(0.6, 0.6)),
            (Coordinate(0.6, 0.6), Coordinate(0.5, 0.4)),
            (Coordinate(0.5, 0.4), Coordinate(0.4, 0.6)),
        ]
    ]


def test_shape_to_geojson_structure():
    circle = GeoCircle(Coordinate(0.0, 0.0), 1000, dt=default_test_datetime)
    # Assert kwargs and properties end up in the right place
    assert circle.to_geojson(properties={'test_prop': 1}, test_kwarg=2) == {
        'type': 'Feature',
        'geometry': {
            'type': 'Polygon',
            'coordinates': [[list(x.to_float()) for x in circle.bounding_coords()]],
        },
        'properties': {
            'test_prop': 1,
            'datetime_start': default_test_datetime.isoformat(),
            'datetime_end': default_test_datetime.isoformat(),
        },
        'test_kwarg': 2,
    }

    # Assert k works as intended
    assert circle.to_geojson(k=10) == {
        'type': 'Feature',
        'geometry': {
            'type': 'Polygon',
            'coordinates': [[list(x.to_float()) for x in circle.bounding_coords(k=10)]],
        },
        'properties': {
            'datetime_start': default_test_datetime.isoformat(),
            'datetime_end': default_test_datetime.isoformat(),
        }
    }

    assert circle.to_geojson()['geometry'] == circle.__geo_interface__


def test_simpleshape_from_fastkml_placemark():
    expected = MultiGeoPolygon(
        [
            GeoBox(Coordinate(0., 1.), Coordinate(1., 0.)).to_polygon()
        ],
        dt=datetime(2020, 1, 1),
        properties={'test': 'prop'}
    )

    placemark = Placemark(
        geometry=MultiPolygon(
            [
                [
                    [x.to_float() for x in ring]
                    for ring in shape
                ] for shape in expected.linear_rings()
            ]
        ),
        times=TimeStamp(timestamp=KmlDateTime(datetime(2020, 1, 1, tzinfo=timezone.utc))),
        extended_data=ExtendedData(elements=[Data(name='test', value='prop')])
    )
    actual = MultiGeoPolygon.from_fastkml_placemark(placemark)
    assert expected == actual

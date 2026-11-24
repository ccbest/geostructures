from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from geostructures import (
    Coordinate, GeoBox, GeoCircle, GeoEllipse, GeoLineString, GeoPoint, GeoPolygon, GeoRing,
)
from geostructures.multistructures import MultiGeoLineString, MultiGeoPoint, MultiGeoPolygon
from geostructures.time import TimeInterval
from geostructures._polars_io import collection_to_html, shapes_to_polars


_SQUARE = [Coordinate(0.0, 0.0), Coordinate(1.0, 0.0), Coordinate(1.0, 1.0), Coordinate(0.0, 0.0)]


def _point(**kwargs):
    return GeoPoint(Coordinate(1.0, 2.0), **kwargs)


def test_column_time_converted_to_utc():
    plus_three = timezone(timedelta(hours=3))
    shapes = [
        _point(dt=datetime(2024, 1, 1, 3, tzinfo=plus_three)),
        _point(dt=datetime(2024, 1, 1, 1, tzinfo=timezone.utc)),
        _point(),
    ]
    frame = shapes_to_polars(shapes)

    expected = [datetime(2024, 1, 1, 0, tzinfo=timezone.utc), datetime(2024, 1, 1, 1, tzinfo=timezone.utc), None]
    for name in ('datetime_start', 'datetime_end'):
        assert frame.schema[name] == pl.Datetime('us', 'UTC')
        assert frame[name].to_list() == expected


def test_column_mixed_types_fall_back_to_object():
    shapes = [_point(properties={'a': 1, 'b': 1}), _point(properties={'a': 'x', 'b': 2})]
    frame = shapes_to_polars(shapes)

    assert frame.schema['a'] == pl.Object
    assert frame['a'].to_list() == [1, 'x']
    assert frame.schema['b'] == pl.Int64


def test_shapes_to_polars_row_order():
    shapes = [_point(properties={'i': i}) for i in (3, 1, 2)]
    assert shapes_to_polars(shapes)['i'].to_list() == [3, 1, 2]


def test_shapes_to_polars_default_columns():
    shapes = [
        _point(properties={'b': 1, 'a': 2}),
        _point(dt=TimeInterval(datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 1, 2, tzinfo=timezone.utc)),
               properties={'c': 3, 'a': 4}),
    ]
    frame = shapes_to_polars(shapes)
    assert frame.columns == ['geometry', 'datetime_start', 'datetime_end', 'b', 'a', 'c']

    # No time columns when no shape has a time
    frame = shapes_to_polars([_point(properties={'b': 1})])
    assert frame.columns == ['geometry', 'b']


def test_shapes_to_polars_include_properties():
    shapes = [
        _point(dt=datetime(2024, 1, 1, tzinfo=timezone.utc), properties={'b': 1, 'a': 2, 'c': 3}),
    ]
    frame = shapes_to_polars(shapes, include_properties=['c', 'datetime_end', 'a', 'missing'])
    assert frame.columns == ['geometry', 'c', 'datetime_end', 'a', 'missing']
    assert frame.schema['datetime_end'] == pl.Datetime('us', 'UTC')
    assert frame['missing'].to_list() == [None]

    assert shapes_to_polars(shapes, include_properties=[]).columns == ['geometry']


def test_shapes_to_polars_missing_key_is_null():
    shapes = [_point(properties={'a': 1}), _point(properties={'b': 'x'})]
    frame = shapes_to_polars(shapes)
    assert frame['a'].to_list() == [1, None]
    assert frame['b'].to_list() == [None, 'x']


def test_shapes_to_polars_geometry_label():
    shapes = [
        GeoPolygon(_SQUARE),
        _point(),
        GeoBox(Coordinate(0.0, 1.0), Coordinate(1.0, 0.0)),
        GeoCircle(Coordinate(0.0, 0.0), 1000),
        GeoEllipse(Coordinate(0.0, 0.0), 1000, 500, 45),
        GeoRing(Coordinate(0.0, 0.0), inner_radius=500, outer_radius=1000),
        GeoLineString([Coordinate(0.0, 0.0), Coordinate(1.0, 1.0)]),
        MultiGeoPoint([_point(), GeoPoint(Coordinate(0.0, 0.0))]),
        MultiGeoLineString([GeoLineString([Coordinate(0.0, 0.0), Coordinate(1.0, 1.0)])]),
        MultiGeoPolygon([GeoPolygon(_SQUARE)]),
        _point(),
    ]
    frame = shapes_to_polars(shapes)
    expected = [
        '<GeoPolygon>', '<GeoPoint>', '<GeoBox>', '<GeoCircle>', '<GeoEllipse>', '<GeoRing>',
        '<GeoLineString>', '<MultiGeoPoint>', '<MultiGeoLineString>', '<MultiGeoPolygon>', '<GeoPoint>',
    ]
    assert frame['geometry'].to_list() == expected
    assert frame.schema['geometry'] == pl.Enum(sorted(set(expected)))
    assert frame.filter(pl.col('geometry') == '<GeoPoint>').height == 2


def test_shapes_to_polars_geometry_shape():
    shapes = [_point(properties={'a': 1}), GeoPolygon(_SQUARE, properties={'a': 2})]
    frame = shapes_to_polars(shapes, geometry='shape')

    assert frame.schema['geometry'] == pl.Object
    assert all(a is b for a, b in zip(frame['geometry'].to_list(), shapes))
    assert frame.filter(pl.col('a') == 2)['geometry'].to_list()[0] is shapes[1]


def test_shapes_to_polars_geometry_invalid():
    with pytest.raises(ValueError):
        shapes_to_polars([_point()], geometry='wkt')


def test_shapes_to_polars_lazy():
    frame = shapes_to_polars([_point(properties={'a': 1})], lazy=True)
    assert isinstance(frame, pl.LazyFrame)
    assert frame.collect()['a'].to_list() == [1]


def test_shapes_to_polars_empty():
    for geometry in ('label', 'shape'):
        frame = shapes_to_polars([], geometry=geometry)
        assert frame.columns == ['geometry']
        assert frame.height == 0


def test_shapes_to_polars_geometry_property():
    # As in to_geopandas, the geometry column replaces a property named 'geometry'
    shapes = [_point(properties={'geometry': 'x', 'a': 1})]
    assert shapes_to_polars(shapes).columns == ['geometry', 'a']
    assert shapes_to_polars(shapes)['geometry'].to_list() == ['<GeoPoint>']
    assert shapes_to_polars(shapes, include_properties=['geometry', 'a']).columns == ['geometry', 'a']


def test_collection_to_html():
    shapes = [
        _point(properties={'a': 1, 'geometry': 'x'}),
        GeoPolygon(_SQUARE, properties={'b': 'y'}),
        _point(dt=datetime(2024, 1, 1, tzinfo=timezone.utc)),
    ]
    out = collection_to_html('FeatureCollection', shapes)
    assert out.startswith('<p>FeatureCollection: 3 shapes x 4 fields (GeoPoint, GeoPolygon)</p>')
    assert '&lt;GeoPoint&gt;' in out and '&lt;GeoPolygon&gt;' in out
    assert 'showing' not in out

    out = collection_to_html('Track', [_point()])
    assert out.startswith('<p>Track: 1 shape x 0 fields (GeoPoint)</p>')

    out = collection_to_html('FeatureCollection', [])
    assert out.startswith('<p>FeatureCollection: 0 shapes x 0 fields</p>')


def test_collection_to_html_long(monkeypatch):
    import geostructures._polars_io as polars_io

    shapes = [_point(properties={'i': i}) for i in range(10_000)]
    converted = []
    original = polars_io.shapes_to_polars

    def _spy(sample, *args, **kwargs):
        converted.append(len(sample))
        return original(sample, *args, **kwargs)

    monkeypatch.setattr(polars_io, 'shapes_to_polars', _spy)
    out = collection_to_html('FeatureCollection', shapes)

    assert converted == [10]
    assert out.startswith(
        '<p>FeatureCollection: 10,000 shapes x 1 field (GeoPoint); showing the first 5 and last 5</p>'
    )
    for i in (0, 4, 9995, 9999):
        assert f'<td>{i}</td>' in out
    assert '<td>5</td>' not in out and '<td>9994</td>' not in out

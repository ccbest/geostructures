
from collections import Counter
from datetime import datetime, timezone
from io import BytesIO, StringIO
from pathlib import Path
from zipfile import ZipFile

import pytest

from geostructures import *
from geostructures.parsers import parse_geojson, parse_kml, parse_shapefile, parse_wkt
from geostructures.serializers import *
from tests.functions import assert_shape_equivalence


_DT = datetime(2020, 1, 1, tzinfo=timezone.utc)


def _sample_collection():
    """A FeatureCollection covering points, lines, polygons w/ holes and multis."""
    return FeatureCollection([
        GeoPoint(Coordinate(1., 2.), dt=_DT, properties={'name': 'pt', 'k': 'v'}),
        GeoLineString(
            [Coordinate(0., 0.), Coordinate(1., 1.), Coordinate(2., 0.)],
            properties={'kind': 'line'},
        ),
        GeoPolygon(
            [Coordinate(0., 0.), Coordinate(0., 3.), Coordinate(3., 3.),
             Coordinate(3., 0.), Coordinate(0., 0.)],
            holes=[GeoPolygon([
                Coordinate(1., 1.), Coordinate(1., 2.), Coordinate(2., 2.),
                Coordinate(2., 1.), Coordinate(1., 1.),
            ])],
            properties={'shape': 'poly'},
        ),
        MultiGeoPoint([GeoPoint(Coordinate(5., 5.)), GeoPoint(Coordinate(6., 6.))]),
    ])


def _assert_round_trips(original, parsed):
    """
    Geometry + dt equivalence plus preservation of the original properties.
    The reader adds provenance keys (``sub_folder_N``, ``filepath``/``filename``),
    so the original properties must be a *subset* of the parsed ones.
    """
    assert isinstance(parsed, FeatureCollection)
    assert len(original) == len(parsed)
    for orig, back in zip(original.geoshapes, parsed.geoshapes):
        assert_shape_equivalence(orig, back)
        for key, value in orig._properties.items():
            assert back._properties[key] == value


def test_serialize_kml_disk_round_trip(kml_round_trip):
    original = _sample_collection()
    _assert_round_trips(original, kml_round_trip(original))


def test_serialize_kmz_disk_round_trip(kml_round_trip, tmp_path):
    original = _sample_collection()
    _assert_round_trips(original, kml_round_trip(original, '.kmz'))

    # The written .kmz is a real zip archive containing the conventional doc.kml
    path = tmp_path / 'test.kmz'
    assert path.read_bytes().startswith(b'PK\x03\x04')
    with ZipFile(path) as archive:
        assert archive.namelist() == ['doc.kml']


def test_serialize_kml_in_memory_bytes():
    original = _sample_collection()

    kml_bytes = serialize_kml(original)
    assert isinstance(kml_bytes, bytes)
    assert kml_bytes.lstrip().startswith(b'<')
    _assert_round_trips(original, parse_kml(kml_bytes))

    kmz_bytes = serialize_kml(original, kmz=True)
    assert kmz_bytes.startswith(b'PK\x03\x04')
    _assert_round_trips(original, parse_kml(kmz_bytes))


def test_serialize_kml_extension_inference(tmp_path):
    original = _sample_collection()

    kml_path = tmp_path / 'out.kml'
    kmz_path = tmp_path / 'out.kmz'
    serialize_kml(original, kml_path)
    serialize_kml(original, kmz_path)

    assert kml_path.read_bytes().lstrip().startswith(b'<')   # plain KML
    assert kmz_path.read_bytes().startswith(b'PK\x03\x04')   # zipped KMZ


def test_serialize_kml_kmz_kwarg_overrides_extension(tmp_path):
    original = _sample_collection()

    # .kml suffix but kmz=True -> zipped despite the extension
    zipped = tmp_path / 'override.kml'
    serialize_kml(original, zipped, kmz=True)
    assert zipped.read_bytes().startswith(b'PK\x03\x04')

    # .kmz suffix but kmz=False -> plain KML despite the extension
    plain = tmp_path / 'override.kmz'
    serialize_kml(original, plain, kmz=False)
    assert plain.read_bytes().lstrip().startswith(b'<')


def test_serialize_kml_accepts_string_path(tmp_path):
    original = _sample_collection()
    path = tmp_path / 'as_str.kml'
    assert serialize_kml(original, str(path)) is None
    _assert_round_trips(original, parse_kml(path))


def test_serialize_kml_single_shape():
    shape = GeoPoint(Coordinate(3., 4.), properties={'solo': 'true'})
    parsed = parse_kml(serialize_kml(shape))
    assert len(parsed) == 1
    assert_shape_equivalence(shape, parsed[0])
    assert parsed[0]._properties['solo'] == 'true'


def test_serialize_kml_track_round_trip(kml_round_trip):
    track = Track([
        GeoPoint(Coordinate(5., 6.), dt=_DT),
        GeoPoint(Coordinate(7., 8.), dt=datetime(2020, 1, 2, tzinfo=timezone.utc)),
    ])
    parsed = kml_round_trip(track, '.kmz')
    assert len(parsed) == 2
    assert {s.dt for s in parsed.geoshapes} == {s.dt for s in track.geoshapes}


def test_serialize_kml_empty_collection(kml_round_trip):
    parsed = kml_round_trip(FeatureCollection([]))
    assert isinstance(parsed, FeatureCollection)
    assert len(parsed) == 0


def test_serialize_kml_to_file_like():
    original = _sample_collection()
    buffer = BytesIO()
    assert serialize_kml(original, buffer) is None
    assert buffer.getvalue().lstrip().startswith(b'<')
    _assert_round_trips(original, parse_kml(buffer.getvalue()))


def test_serialize_kml_custom_folder_name(tmp_path):
    original = FeatureCollection([GeoPoint(Coordinate(1., 1.))])
    parsed = parse_kml(serialize_kml(original, folder_name='my_folder'))
    # parse_kml records the KML Folder name as a sub_folder_N property
    assert parsed[0]._properties['sub_folder_2'] == 'my_folder'


def test_serialize_kml_invalid_file_raises():
    with pytest.raises(TypeError):
        serialize_kml(_sample_collection(), 123)


# --------------------------------------------------------------------------- #
# GeoJSON
# --------------------------------------------------------------------------- #

def test_serialize_geojson_collection_round_trip():
    original = _sample_collection()
    text = serialize_geojson(original)
    assert isinstance(text, str)

    parsed = parse_geojson(text)
    assert isinstance(parsed, FeatureCollection)
    assert len(parsed) == len(original)
    for orig, back in zip(original.geoshapes, parsed.geoshapes):
        assert_shape_equivalence(orig, back)


def test_serialize_geojson_single_shape():
    shape = GeoPoint(Coordinate(1., 2.), properties={'a': 'b'})
    parsed = parse_geojson(serialize_geojson(shape))
    assert isinstance(parsed, GeoPoint)
    assert_shape_equivalence(shape, parsed)
    assert parsed.properties['a'] == 'b'


def test_serialize_geojson_indent_and_kwargs():
    fc = FeatureCollection([GeoPoint(Coordinate(1., 2.))])
    # indent forces multi-line pretty output
    assert '\n' in serialize_geojson(fc, indent=2)
    # kwargs are forwarded to to_geojson (extra properties merged in)
    text = serialize_geojson(fc, properties={'injected': 'yes'})
    assert parse_geojson(text)[0].properties['injected'] == 'yes'


def test_serialize_geojson_to_path_and_file_like(tmp_path):
    original = _sample_collection()

    path = tmp_path / 'out.geojson'
    assert serialize_geojson(original, path) is None
    assert len(parse_geojson(path.read_text())) == len(original)

    buffer = StringIO()
    assert serialize_geojson(original, buffer) is None
    assert len(parse_geojson(buffer.getvalue())) == len(original)


# --------------------------------------------------------------------------- #
# WKT
# --------------------------------------------------------------------------- #

def test_serialize_wkt_single_shape_round_trip():
    shape = GeoPolygon([
        Coordinate(0., 0.), Coordinate(0., 3.), Coordinate(3., 3.),
        Coordinate(3., 0.), Coordinate(0., 0.),
    ])
    text = serialize_wkt(shape)
    assert isinstance(text, str) and text.startswith('POLYGON')
    assert_shape_equivalence(shape, parse_wkt(text))


def test_serialize_wkt_multishape():
    multi = MultiGeoPoint([GeoPoint(Coordinate(1., 1.)), GeoPoint(Coordinate(2., 2.))])
    text = serialize_wkt(multi)
    assert text.startswith('MULTIPOINT')
    assert_shape_equivalence(multi, parse_wkt(text))


def test_serialize_wkt_to_path_and_file_like(tmp_path):
    shape = GeoLineString([Coordinate(0., 0.), Coordinate(1., 1.)])

    path = tmp_path / 'out.wkt'
    assert serialize_wkt(shape, path) is None
    assert_shape_equivalence(shape, parse_wkt(path.read_text()))

    buffer = StringIO()
    assert serialize_wkt(shape, buffer) is None
    assert buffer.getvalue().startswith('LINESTRING')


def test_serialize_wkt_rejects_collection():
    with pytest.raises(TypeError):
        serialize_wkt(_sample_collection())


def test_serialize_text_invalid_file_raises():
    shape = GeoPoint(Coordinate(1., 2.))
    with pytest.raises(TypeError):
        serialize_geojson(shape, 123)
    with pytest.raises(TypeError):
        serialize_wkt(shape, 123)


# --------------------------------------------------------------------------- #
# Shapefile
# --------------------------------------------------------------------------- #

def test_serialize_shapefile_in_memory_bytes():
    payload = serialize_shapefile(_sample_collection())
    assert isinstance(payload, bytes)
    assert payload.startswith(b'PK\x03\x04')          # a real zip archive
    with ZipFile(BytesIO(payload)) as archive:
        # pyshp emits the standard shapefile component set
        assert any(name.endswith('.shp') for name in archive.namelist())


def test_serialize_shapefile_disk_round_trip(tmp_path):
    original = _sample_collection()
    path = tmp_path / 'out.zip'
    assert serialize_shapefile(original, path) is None

    parsed = parse_shapefile(path)
    assert len(parsed) == len(original)
    # Shapefiles split by geometry family, so compare type multisets not order
    assert (Counter(type(s).__name__ for s in parsed.geoshapes)
            == Counter(type(s).__name__ for s in original.geoshapes))


def test_serialize_shapefile_single_shape(tmp_path):
    shape = GeoPoint(Coordinate(3., 4.))
    path = tmp_path / 'single.zip'
    serialize_shapefile(shape, path)
    parsed = parse_shapefile(path)
    assert len(parsed) == 1
    assert isinstance(parsed[0], GeoPoint)


def test_serialize_shapefile_to_file_like():
    buffer = BytesIO()
    assert serialize_shapefile(_sample_collection(), buffer) is None
    assert buffer.getvalue().startswith(b'PK\x03\x04')


def test_serialize_shapefile_include_properties(tmp_path):
    fc = FeatureCollection([GeoPoint(Coordinate(1., 2.), properties={'keep': 'x', 'drop': 'y'})])
    path = tmp_path / 'props.zip'
    serialize_shapefile(fc, path, include_properties=['keep'])
    props = parse_shapefile(path)[0].properties
    assert 'keep' in props
    assert 'drop' not in props


def test_serialize_shapefile_invalid_file_raises():
    with pytest.raises(TypeError):
        serialize_shapefile(_sample_collection(), 123)


from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from geostructures import *
from geostructures.parsers import *
from geostructures.time import TimeInterval
from tests.functions import assert_geopolygons_equal, assert_geolinestrings_equal, \
    assert_geopoints_equal, assert_multishapes_equal


def test_parse_kml_fastkml_object():
    import fastkml

    folder = fastkml.Folder(name='test folder')
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
    parsed = parse_kml(folder)
    assert isinstance(parsed, FeatureCollection)
    assert list(parsed) == [
        GeoPoint(
            Coordinate(1., 0.),
            dt=datetime(2020, 1, 1),
        )
    ]
    assert parsed[0]._properties == {
        'test': 'prop',
        'sub_folder_0': 'test folder',
        'name': 'test name',
        'description': 'test description'
    }

    # A bare Placemark is also accepted
    placemark = GeoPoint(Coordinate(1., 0.)).to_fastkml_placemark(name='solo')
    solo = parse_kml(placemark)
    assert isinstance(solo, FeatureCollection)
    assert solo[0].properties['name'] == 'solo'


def test_parse_kml_schemadata():
    # Assert SchemaData fields get parsed correctly, from a raw KML string
    with open('./tests/test_files/test_schemadata.kml', 'r') as f:
        kml_str = f.read()

    result = parse_kml(kml_str)
    assert isinstance(result, FeatureCollection)
    assert result[0].properties['TrailHeadName'] == 'Pi in the sky'


def test_parse_kml_mixed_extended_data():
    # Regression: a placemark whose ExtendedData mixes a SchemaData element and
    # a plain Data element must retain properties from both.
    result = parse_kml('./tests/test_files/test_mixed_extendeddata.kml')
    props = result[0].properties
    assert props['TrailHeadName'] == 'Pi in the sky'  # from SchemaData
    assert props['holler'] == 'world'                 # from Data


def test_parse_kml_input_forms():
    import fastkml

    with open('./tests/test_files/test_kml.kml', 'r', encoding='utf8') as f:
        kml_str = f.read()

    # Raw KML string, raw KML bytes, and a pre-parsed fastkml object should all
    # agree, and each should return a FeatureCollection. fastkml.KML.from_string
    # is handed bytes because lxml rejects str carrying an encoding declaration
    from_str = parse_kml(kml_str)
    from_bytes = parse_kml(kml_str.encode('utf8'))
    from_obj = parse_kml(fastkml.KML.from_string(kml_str.encode('utf8')))

    for result in (from_str, from_bytes, from_obj):
        assert isinstance(result, FeatureCollection)
        assert len(result) == 19

    with pytest.raises(TypeError):
        parse_kml(1234)


def test_parse_kml_encoding_declaration_and_bom():
    # Regression: with lxml installed (fastkml's preferred XML backend), str
    # input containing an XML encoding declaration raised ValueError. A UTF-8
    # byte order mark also misrouted str input to the filepath branch.
    with open('./tests/test_files/test_kml.kml', 'r', encoding='utf8') as f:
        kml_str = f.read()
    assert kml_str.startswith('<?xml')

    assert len(parse_kml('\ufeff' + kml_str)) == 19
    assert len(parse_kml(b'\xef\xbb\xbf' + kml_str.encode('utf8'))) == 19


def test_parse_kml_linearring():
    # Regression: KML allows a LinearRing as a direct placemark geometry;
    # it was previously dropped silently. It parses as the enclosed polygon.
    kml_str = '''
        <kml xmlns="http://www.opengis.net/kml/2.2"><Document><Placemark>
        <LinearRing><coordinates>0,0 1,0 1,1 0,0</coordinates></LinearRing>
        </Placemark></Document></kml>
    '''
    parsed = parse_kml(kml_str)
    assert len(parsed) == 1
    assert parsed[0] == GeoPolygon([
        Coordinate(0., 0.), Coordinate(1., 0.), Coordinate(1., 1.), Coordinate(0., 0.),
    ])


def test_parse_kml_open_ended_timespan():
    # Regression: KML allows TimeSpans with only a <begin> or only an <end>;
    # these previously raised AttributeError and aborted the whole parse.
    # They become zero-length intervals at the known endpoint.
    template = '''
        <kml xmlns="http://www.opengis.net/kml/2.2"><Document><Placemark>
        <TimeSpan>{}</TimeSpan>
        <Point><coordinates>1.0,2.0</coordinates></Point>
        </Placemark></Document></kml>
    '''

    begin_only = parse_kml(template.format('<begin>2020-01-01T00:00:00Z</begin>'))
    assert begin_only[0].dt == TimeInterval(
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        datetime(2020, 1, 1, tzinfo=timezone.utc),
    )

    end_only = parse_kml(template.format('<end>2021-06-15T12:00:00Z</end>'))
    assert end_only[0].dt == TimeInterval(
        datetime(2021, 6, 15, 12, tzinfo=timezone.utc),
        datetime(2021, 6, 15, 12, tzinfo=timezone.utc),
    )

    empty = parse_kml(template.format(''))
    assert empty[0].dt is None


def test_parse_geojson():
    # No need to test properties or times - tested elsewhere
    shape = GeoCircle(Coordinate(0., 0.), 500).to_polygon()
    assert parse_geojson(shape.to_geojson()) == shape

    # Stringified geojson
    assert parse_geojson(json.dumps(shape.to_geojson())) == shape

    # Just the geo interface
    assert parse_geojson(shape.to_geojson()['geometry']) == shape


def test_parse_wkt():
    shape = GeoCircle(Coordinate(0., 0.), 500).to_polygon()
    assert_geopolygons_equal(parse_wkt(shape.to_wkt()), shape)

    shape = GeoLineString([Coordinate(0., 0.), Coordinate(0., 0.)])
    assert_geolinestrings_equal(parse_wkt(shape.to_wkt()), shape)

    shape = GeoPoint(Coordinate(1., 0.))
    assert_geopoints_equal(parse_wkt(shape.to_wkt()), shape)

    shape = MultiGeoPolygon([
        GeoCircle(Coordinate(0., 0.), 500).to_polygon(),
        GeoCircle(Coordinate(0., 0.), 500).to_polygon()
    ])
    assert_multishapes_equal(parse_wkt(shape.to_wkt()), shape)

    shape = MultiGeoLineString([
        GeoLineString([Coordinate(0., 0.), Coordinate(0., 0.)]),
        GeoLineString([Coordinate(0., 0.), Coordinate(0., 0.)])
    ])
    assert_multishapes_equal(parse_wkt(shape.to_wkt()), shape)

    shape = MultiGeoPoint([
        GeoPoint(Coordinate(1., 0.)),
        GeoPoint(Coordinate(1., 0.))
    ])
    assert_multishapes_equal(parse_wkt(shape.to_wkt()), shape)

    shape = GeoPoint(Coordinate(1., 0., z=50., m=10.))
    assert_geopoints_equal(parse_wkt(shape.to_wkt()), shape)

    with pytest.raises(ValueError):
        parse_wkt('123')

    with pytest.raises(ValueError):
        parse_wkt('worse')


def test_parse_kml_folder_traceability():
    import fastkml

    # A document containing [Folder A [placemark], propertyless placemark]
    document = fastkml.Document(name='doc')
    folder_a = fastkml.Folder(name='folder A')
    folder_a.append(
        GeoPoint(Coordinate(1., 0.)).to_fastkml_placemark(name='inside A')
    )
    document.append(folder_a)
    document.append(
        GeoPoint(Coordinate(2., 0.)).to_fastkml_placemark(name='outside A')
    )

    parsed = parse_kml(document)
    by_name = {shape.properties['name']: shape for shape in parsed}

    # Folder names apply to shapes inside the folder, even when the
    # placemark carries no properties of its own
    assert by_name['inside A'].properties['sub_folder_1'] == 'folder A'

    # ...and must not leak to siblings outside the folder
    assert 'sub_folder_1' not in by_name['outside A'].properties


def test_parse_kml_files():
    mock_kml = parse_kml('./tests/test_files/test_kml.kml')
    assert isinstance(mock_kml, FeatureCollection)
    assert len(mock_kml) == 19
    # Plain KML shapes are tagged with both filepath and filename
    assert mock_kml[0].properties['filepath'] == 'tests/test_files/test_kml.kml'
    assert mock_kml[0].properties['filename'] == 'test_kml.kml'

    # A pathlib.Path is accepted as well as a str
    mock_kmz = parse_kml(Path('./tests/test_files/test_kmz.kmz'))
    assert len(mock_kmz) == 83
    assert mock_kmz[0].properties['filepath'] == str(Path('./tests/test_files/test_kmz.kmz'))
    assert mock_kmz[0].properties['filename'].lower().endswith('.kml')

    with pytest.raises(FileNotFoundError):
        parse_kml('./bad_path.kml')


def test_parse_kml_skips_unsupported_geometry():
    import fastkml
    import pygeoif

    # A heterogeneous MultiGeometry surfaces as a GeometryCollection, which has
    # no single geostructures equivalent - it should be skipped, not raise.
    collection = pygeoif.geometry.GeometryCollection([
        pygeoif.geometry.Point(0, 0),
        pygeoif.geometry.LineString([(0, 0), (1, 1)]),
    ])
    document = fastkml.Document(name='doc')
    document.append(fastkml.Placemark(geometry=collection))
    document.append(GeoPoint(Coordinate(1., 0.)).to_fastkml_placemark(name='ok'))

    parsed = parse_kml(document)
    # The unsupported placemark is dropped; the valid one survives
    assert len(parsed) == 1
    assert parsed[0].properties['name'] == 'ok'


def test_parse_kml_kmz_bytes():
    # Raw KMZ archive bytes are detected via the zip magic number
    kmz_bytes = Path('./tests/test_files/test_kmz.kmz').read_bytes()
    result = parse_kml(kmz_bytes)
    assert isinstance(result, FeatureCollection)
    assert len(result) == 83
    # No filepath for raw bytes (there is no path), but members are named
    assert 'filepath' not in result[0].properties
    assert result[0].properties['filename'].lower().endswith('.kml')


def test_parse_shapefile_delegates_to_from_shapefile(tmp_path):
    from zipfile import ZipFile

    fc = FeatureCollection([
        GeoPoint(Coordinate(1., 2.), properties={'a': 'b'}),
        GeoPoint(Coordinate(3., 4.), properties={'a': 'c'}),
    ])
    zip_path = tmp_path / 'points.zip'
    with ZipFile(zip_path, 'w') as zfile:
        fc.to_shapefile(zfile)

    parsed = parse_shapefile(zip_path)
    assert isinstance(parsed, FeatureCollection)
    # parse_shapefile is a thin delegate, so it must match from_shapefile exactly
    assert parsed == FeatureCollection.from_shapefile(zip_path)
    assert len(parsed) == 2


def test_parse_shapefile_forwards_read_layers(tmp_path):
    from zipfile import ZipFile

    fc = FeatureCollection([GeoPoint(Coordinate(1., 2.))])
    zip_path = tmp_path / 'points.zip'
    with ZipFile(zip_path, 'w') as zfile:
        fc.to_shapefile(zfile)

    # A non-existent layer filter yields an empty collection (args are forwarded)
    assert len(parse_shapefile(zip_path, read_layers=['does_not_exist'])) == 0


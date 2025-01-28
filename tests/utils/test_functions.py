
from datetime import datetime, timezone

from geostructures.utils.functions import *
from geostructures.time import TimeInterval


def test_default_to_zulu(caplog):
    dt = datetime(2020, 1, 1, tzinfo=timezone.utc)
    assert default_to_zulu(dt).tzinfo == timezone.utc

    dt = datetime(2020, 1, 1)
    assert default_to_zulu(dt).tzinfo == timezone.utc


# def test_get_dt_from_geojson_props():
#     props = {
#         'datetime_start': '2020-01-01 00:00:00',
#         'datetime_end': '2020-01-01 00:00:00',
#     }
#     assert get_dt_from_geojson_props(props) == TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 1))

#     props = {}
#     assert get_dt_from_geojson_props(props) is None

#     props = {
#         'datetime_start': '2020-01-01 00:00:00',
#     }
#     assert get_dt_from_geojson_props(props) == datetime(2020, 1, 1)

#     props = {
#         'start': '2020-01-01',
#         'end': '2020-01-01',
#     }
#     assert get_dt_from_geojson_props(props, time_start_field='start', time_end_field='end', time_format='%Y-%m-%d') == TimeInterval(datetime(2020, 1, 1), datetime(2020, 1, 1))


def test_round_half_up():
    assert round_half_up(1.59, 1) == 1.6
    assert round_half_up(1.51, 1) == 1.5
    assert round_half_up(1.55, 1) == 1.6
    assert round_half_up(1.65, 1) == 1.7

    assert round_half_up(-1.59, 1) == -1.6
    assert round_half_up(-1.51, 1) == -1.5
    assert round_half_up(-1.55, 1) == -1.5
    assert round_half_up(-1.65, 1) == -1.6


def test_sanitize_json():
    val = {
        '1': datetime(2020, 1, 1),
        '2': {
            'test': datetime(2020, 1, 1),
            'test2': 'test'
        },
        '3': [
            datetime(2020, 1, 1)
        ]
    }
    assert sanitize_json(val) == {
        '1': '2020-01-01T00:00:00',
        '2': {
            'test': '2020-01-01T00:00:00',
            'test2': 'test'
        },
        '3': ['2020-01-01T00:00:00']
    }


def test_is_sub_list():
    assert is_sub_list(
        [1, 2],
        [1, 2, 3]
    )
    assert is_sub_list(
        [1, 2],
        [0, 1, 2, 3]
    )
    assert not is_sub_list(
        [0, 1, 2, 3],
        [1, 2]
    )
    assert not is_sub_list(
        ['a', 'b'],
        [1, 2, 3]
    )

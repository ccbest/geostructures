
from geostructures.utils.functions import *


def test_float_to_str():
    assert float_to_str(1.02) == '1.02'
    assert float_to_str(0.00000000000000000001) == '0.00000000000000000001'
    assert float_to_str(100000000000000000000.) == '100000000000000000000'


def test_round_half_up():
    assert round_half_up(1.59, 1) == 1.6
    assert round_half_up(1.51, 1) == 1.5
    assert round_half_up(1.55, 1) == 1.6
    assert round_half_up(1.65, 1) == 1.7

    assert round_half_up(-1.59, 1) == -1.6
    assert round_half_up(-1.51, 1) == -1.5
    assert round_half_up(-1.55, 1) == -1.5
    assert round_half_up(-1.65, 1) == -1.6

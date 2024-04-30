
from geostructures.utils.functions import *

def test_round_half_up():
    assert round_half_up(1.59, 1) == 1.6
    assert round_half_up(1.51, 1) == 1.5
    assert round_half_up(1.55, 1) == 1.6
    assert round_half_up(1.65, 1) == 1.7

    assert round_half_up(-1.59, 1) == -1.6
    assert round_half_up(-1.51, 1) == -1.5
    assert round_half_up(-1.55, 1) == -1.5
    assert round_half_up(-1.65, 1) == -1.6


import re

from geostructures.utils.logging import warn_once


def test_warn_once(caplog):
    warn_once('test')
    assert 'test' in caplog.text

    warn_once('test')
    assert len(re.findall('test', caplog.text)) == 1

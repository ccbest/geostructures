
import re

from geostructures.utils.mixins import LoggingMixin


class Foo(LoggingMixin):
    pass


class Bar(LoggingMixin):
    pass


def test_warn_once(caplog):
    foo = Foo()
    foo.warn_once('test %s', 'test')
    assert 'test test' in caplog.text

    bar = Bar()
    bar.warn_once('test %s', 'test')
    assert len(re.findall('test test', caplog.text)) == 1

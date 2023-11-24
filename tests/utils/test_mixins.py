
import re

from geostructures.utils.mixins import WarnOnceMixin


class Foo(WarnOnceMixin):
    pass


class Bar(WarnOnceMixin):
    pass


def test_warn_once(caplog):
    foo = Foo()
    foo.warn_once('test %s', 'test')
    assert 'test test' in caplog.text

    bar = Bar()
    bar.warn_once('test %s', 'test')
    assert len(re.findall('test test', caplog.text)) == 1


import pytest


def test_compile():
    # Make sure optional dependencies haven't been installed yet
    with pytest.raises(ImportError):
        import fastkml

    # If any optional dependencies are declared in base modules, will raise ImportError
    import geostructures.coordinates
    import geostructures.structures
    import geostructures.multistructures
    import geostructures.calc
    import geostructures.time
    import geostructures.typing
    import geostructures.collections
    import geostructures.parsers

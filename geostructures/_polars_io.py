"""
Conversion of a sequence of shapes to a Polars frame
"""

__all__ = ['shapes_to_polars']

from datetime import timezone
from typing import Any, List, Optional, Sequence

from geostructures._base import BaseShape
from geostructures.utils.conditional_imports import import_optional


_TIME_KEYS = ('datetime_start', 'datetime_end')


def _column(pl: Any, name: str, values: List[Any]):
    """
    Build one property column. Time columns are converted to UTC; a column
    whose values Polars cannot hold in a single type falls back to pl.Object.
    """
    if name in _TIME_KEYS:
        return pl.Series(
            name,
            [None if x is None else x.astimezone(timezone.utc) for x in values],
            dtype=pl.Datetime('us', 'UTC'),
        )

    try:
        return pl.Series(name, values)
    except TypeError:
        return pl.Series(name, values, dtype=pl.Object)


def shapes_to_polars(
    shapes: Sequence[BaseShape],
    include_properties: Optional[List[str]] = None,
    lazy: bool = False,
    geometry: str = 'label',
):
    """
    Convert shapes to a Polars frame with one row per shape, in the order
    given. See CollectionBase.to_polars.

    Args:
        shapes:
            The shapes to convert

        include_properties: (Optional)
            The property keys to include as columns, in this order. Defaults
            to the time columns (if any shape has them), then every other
            property key in the order first seen.

        lazy: (Default False)
            Return a polars.LazyFrame instead of a polars.DataFrame

        geometry: (Default 'label')
            'label' for a pl.Enum column reading e.g. '<GeoPoint>', or
            'shape' for a pl.Object column holding the shapes themselves

    Returns:
        polars.DataFrame, or polars.LazyFrame if lazy is True
    """
    import_optional('polars')
    import polars as pl

    if geometry == 'label':
        labels = [f'<{type(shape).__name__}>' for shape in shapes]
        geom = pl.Series('geometry', labels, dtype=pl.Enum(sorted(set(labels))))
    elif geometry == 'shape':
        geom = pl.Series('geometry', list(shapes), dtype=pl.Object)
    else:
        raise ValueError(f"geometry must be 'label' or 'shape', not {geometry!r}")

    rows = [shape.properties for shape in shapes]
    if include_properties is None:
        seen = list(dict.fromkeys(key for row in rows for key in row))
        keys = [k for k in _TIME_KEYS if k in seen] + [k for k in seen if k not in _TIME_KEYS]
    else:
        keys = list(include_properties)

    # As in to_geopandas, the geometry column takes precedence over a property of the same name
    keys = [key for key in keys if key != 'geometry']

    frame = pl.DataFrame(
        [geom] + [_column(pl, key, [row.get(key) for row in rows]) for key in keys]
    )
    return frame.lazy() if lazy else frame

"""
Conversion of a sequence of shapes to a Polars frame
"""

__all__ = ['collection_to_html', 'shapes_to_polars']

from datetime import timezone
import html
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from geostructures._base import BaseShape
from geostructures._shapely_io import LazyShapes
from geostructures.utils.conditional_imports import import_optional


_TIME_KEYS = ('datetime_start', 'datetime_end')

# Collections longer than this show only their first and last _HTML_EDGE shapes
_HTML_MAX_ROWS = 10
_HTML_EDGE = 5


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


def _default_keys(rows: Sequence[Dict]) -> List[str]:
    """
    The property columns of a frame built without include_properties, given
    each shape's properties: the time columns (if any shape has them), then
    every other key in the order first seen. A property named 'geometry' is
    left out, as the geometry column takes its place.
    """
    seen = list(dict.fromkeys(key for row in rows for key in row))
    keys = [k for k in _TIME_KEYS if k in seen] + [k for k in seen if k not in _TIME_KEYS]
    return [key for key in keys if key != 'geometry']


def _names_and_properties(shapes: Sequence[BaseShape]) -> Tuple[List[str], List[Dict]]:
    """
    Each shape's class name and properties. Shapes a LazyShapes hasn't created
    yet are read from its source data and stay uncreated.
    """
    if not isinstance(shapes, LazyShapes):
        return [type(shape).__name__ for shape in shapes], [shape.properties for shape in shapes]

    created = shapes.created()
    return shapes.type_names(), [
        shapes[i].properties if created[i] else shapes.properties_at(i)
        for i in range(len(shapes))
    ]


def _plural(count: int, noun: str) -> str:
    return f'{count:,} {noun}' + ('' if count == 1 else 's')


def collection_to_html(class_name: str, shapes: Sequence[BaseShape]) -> str:
    """
    Render a collection for notebook display: a header line naming the
    collection, its size and its shape types, then its table. Collections of
    more than ten shapes show only the first five and last five, and only
    those are converted.

    Args:
        class_name:
            The collection's class name, e.g. 'FeatureCollection'

        shapes:
            The collection's shapes

    Returns:
        An HTML string
    """
    import_optional('polars')
    import polars as pl

    names, rows = _names_and_properties(shapes)
    keys = _default_keys(rows)
    shape_types = sorted(set(names))

    header = f'{class_name}: {_plural(len(shapes), "shape")} x {_plural(len(keys), "field")}'
    if shape_types:
        header += f' ({", ".join(shape_types)})'

    sample = shapes
    if len(shapes) > _HTML_MAX_ROWS:
        edges = list(range(_HTML_EDGE)) + list(range(len(shapes) - _HTML_EDGE, len(shapes)))
        if isinstance(shapes, LazyShapes):
            sample = shapes.take(np.array(edges))
        else:
            sample = [shapes[i] for i in edges]
        header += f'; showing the first {_HTML_EDGE} and last {_HTML_EDGE}'

    frame = shapes_to_polars(sample, include_properties=keys)
    with pl.Config(tbl_rows=_HTML_MAX_ROWS):
        table = frame._repr_html_()  # pylint: disable=protected-access

    return f'<p>{html.escape(header)}</p>\n{table}'


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

    names, rows = _names_and_properties(shapes)
    if geometry == 'label':
        labels = [f'<{name}>' for name in names]
        geom = pl.Series('geometry', labels, dtype=pl.Enum(sorted(set(labels))))
    elif geometry == 'shape':
        # The one case that needs the shapes themselves
        geom = pl.Series('geometry', list(shapes), dtype=pl.Object)
    else:
        raise ValueError(f"geometry must be 'label' or 'shape', not {geometry!r}")

    if include_properties is None:
        keys = _default_keys(rows)
    else:
        # As in to_geopandas, the geometry column takes precedence over a property of the same name
        keys = [key for key in include_properties if key != 'geometry']

    frame = pl.DataFrame(
        [geom] + [_column(pl, key, [row.get(key) for row in rows]) for key in keys]
    )
    return frame.lazy() if lazy else frame

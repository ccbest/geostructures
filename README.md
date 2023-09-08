
# Geostructures

A lightweight implementation of shapes drawn across a geo-temporal plane.

![Pull Request Quality Check](https://github.com/ccbest/geostructures/actions/workflows/pr_quality_check.yml/badge.svg)

### Installation

Geostructures is available on PYPI
```
$ pip install geostructures
```

### Overview

For an interactive introduction, please review our collection of [Jupyter notebooks](./notebooks).

Geostructures provides a python interface for functionally defining various shapes drawn on a map. Unlike other libraries such as Shapely, these shapes retain their mathematical definitions rather than being simplified into N-sided polygons.

The shapes currently supported are:
* Boxes
* Circles
* Ellipses
* LineStrings
* Points
* Polygons
* Rings/Wedges

All shapes may optionally be temporally-bound using a specific datetime or a datetime interval.

Additionally, geostructures provides convenience objects for representing chronologically-ordered (`Track`) and unordered (`FeatureCollection`) collections of the above shapes.

### Projections

This library assumes that all geospatial terms and structures conform to the 
[WGS84 standard](https://en.wikipedia.org/wiki/World_Geodetic_System).

### Reporting Issues / Requesting Features

The Geostructures team uses Github issues to track development goals. Please include as much detail as possible so we can effectively triage your request.

### Contributing

We welcome all contributors! Please review [CONTRIBUTING.md](./CONTRIBUTING.md) for more information.

### Developers

Carl Best (Sr. Data Scientist/Project Owner)\
https://github.com/ccbest/

Jessica Moore (Sr. Data Scientist)\
https://github.com/jessica-writes-code


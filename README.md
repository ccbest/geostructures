
# Geostructures

A pure-python implementation of shapes drawn across a temporal-geospatial plane.

### Installation

Pending publication to PYPI

### Overview

For an interactive introduction, please review our collection of [Jupyter notebooks](./notebooks).

Geostructures provides a python interface for functionally defining various shapes drawn on a map. Unlike
other libraries such as Shapely, these shapes retain their mathematical definitions rather than being simplified
into N-sided polygons.

The shapes currently supported are:
* Boxes
* Circles
* Ellipses
* LineStrings
* Points
* Polygons
* Rings/Wedges

All shapes may optionally be temporally-bound using a specific date/time or a date/time interval.

Additionally, geostructures provides convenience objects for representing chronologically-ordered (`Track`) and 
unordered (`FeatureCollection`) collections of the above shapes.

### Projections

This library assumes that all geospatial terms and structures conform to the 
[WGS84 standard](https://en.wikipedia.org/wiki/World_Geodetic_System).

### Reporting Issues / Requesting Features

The Geostructures team uses Github issues to track development goals. Please include as much detail as possible
so we can effectively triage your request.

### Contributing

We welcome all contributors! Please review [CONTRIBUTING.md](./CONTRIBUTING.md) for more information.

### Developers

Carl Best (Sr. Data Scientist/Project Owner)
https://github.com/ccbest/

Jessica Moore (Sr. Data Scientist)
https://github.com/jessica-writes-code


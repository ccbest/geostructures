"""Package build script"""
import os
import re
import setuptools

ver_file = f'geostructures{os.sep}_version.py'
__version__ = None

# Pull package version number from _version.py
with open(ver_file, 'r') as f:
    for line in f.readlines():
        if re.match(r'^\s*#', line):  # comment
            continue

        ver_line = line
        verstr = re.match(r"^.*=\s+'(v\d+\.\d+\.\d+(?:\.[a-zA-Z0-9]+)?)'", ver_line)
        if verstr is not None and len(verstr.groups()) == 1:
            __version__ = verstr.groups()[0]
            break

    if __version__ is None:
        raise EnvironmentError(f'Could not find valid version number in {ver_file}; aborting setup')

with open("./README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="geostructures",
    version=__version__,
    author="Carl Best",
    author_email="",
    description="A lightweight implementation of shapes drawn across a geo-temporal plane.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/ccbest/geostructures",
    packages=setuptools.find_packages(
        include=('geostructures*', ),
        exclude=('*tests', 'tests*')
    ),
    package_data={"geostructures": ["py.typed"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.9',
    install_requires=[
        'numpy>=2,<3',
    ],
    extras_require={
        'df': ['geopandas>=1,<2', 'pandas>=2,<3'],
        'h3': ['h3>=4.1,<5'],
        'kml': ['fastkml>=1,<2'],
        'mgrs': ['mgrs>=1.4.5,<2'],
        'proj': ['pyproj>=3.6,<4'],
        'shapely': ['shapely>=2.0.7'],
        'shapefile': ['pyshp>=2.3.1,<3']
    }
)


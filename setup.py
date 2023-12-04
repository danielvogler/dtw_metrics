"""Setup file."""
from setuptools import setup

from dtwmetrics.version import __version__

setup(
    name="dtwmetrics",
    version=__version__,
    description="Dynamic time warping metrics",
    long_description="Dynamic time warping metrics",
    url="https://github.com/danielvogler/dtw_metrics",
    author="Daniel Vogler",
    author_email="geopard.py@gmail.com",
    license="MIT",
    packages=["dtwmetrics"],
    install_requires=["scipy>=1.5.4"],
)

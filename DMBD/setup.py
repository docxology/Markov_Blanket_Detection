"""Compatibility shim for legacy `python setup.py` invocations.

All packaging metadata lives in ``pyproject.toml`` (PEP 621); setuptools reads
it through the ``setuptools.build_meta`` backend declared there.
"""

from setuptools import setup

setup()

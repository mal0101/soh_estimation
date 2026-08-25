"""Compatibility shim.

All project metadata (dependencies, entry points, packaging) lives in
pyproject.toml. This file exists only so legacy tooling that insists on
setup.py keeps working; ``pip install -e .`` reads pyproject.toml.
"""

from setuptools import setup

if __name__ == "__main__":
    setup()

"""Colcon helper: exposes the pure-Python adapter for local testing.

The production ROS 2 build uses package.xml + CMakeLists (added when the
Colab/GPU workspace is created). This file only documents the Python path so
`pytest` and plain interpreters can import `foveamap_ros` without ROS 2.
"""
from setuptools import find_packages  # noqa: F401

"""Build the _shard_reader C extension in-place.

Run with: python build_ext.py
"""
from setuptools import setup, Extension
import numpy as np

import sys

_extra_link = ["-lpthread"] if sys.platform != "darwin" else []

ext = Extension(
    "annbatch._shard_reader",
    sources=["src/annbatch/_shard_reader.c"],
    include_dirs=[np.get_include()],
    extra_compile_args=["-O3"],
    extra_link_args=_extra_link,
)
setup(
    name="_shard_reader",
    ext_modules=[ext],
    script_args=["build_ext", "--inplace", "--build-lib", "src"],
)

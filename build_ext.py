"""Build the _shard_reader C extension in-place.

Run with: python build_ext.py
"""
from setuptools import setup, Extension
import numpy as np

ext = Extension(
    "annbatch._shard_reader",
    sources=["src/annbatch/_shard_reader.c"],
    include_dirs=[np.get_include()],
    extra_compile_args=["-O3"],
)
setup(
    name="_shard_reader",
    ext_modules=[ext],
    script_args=["build_ext", "--inplace", "--build-lib", "src"],
)

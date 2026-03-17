"""Build the _shard_reader C extension in-place.

Run with: python build_ext.py
"""
from setuptools import setup, Extension
import numpy as np

import platform

_compile_args = ["-O3"]
_link_args = []
if platform.system() != "Darwin":
    _compile_args.append("-pthread")
    _link_args.append("-pthread")

ext = Extension(
    "annbatch._shard_reader",
    sources=["src/annbatch/_shard_reader.c"],
    include_dirs=[np.get_include()],
    extra_compile_args=_compile_args,
    extra_link_args=_link_args,
)
setup(
    name="_shard_reader",
    ext_modules=[ext],
    script_args=["build_ext", "--inplace", "--build-lib", "src"],
)

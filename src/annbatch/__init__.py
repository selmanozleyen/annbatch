from __future__ import annotations

from importlib.metadata import version

from . import types
from .io import add_to_collection, create_anndata_collection, write_sharded
from .loader import Loader
from .sampler import CategorySampler, MaskedSampler, RangeSampler, Sampler, SliceSampler

__version__ = version("annbatch")

__all__ = [
    "CategorySampler",
    "Loader",
    "MaskedSampler",
    "RangeSampler",
    "Sampler",
    "SliceSampler",
    "add_to_collection",
    "create_anndata_collection",
    "types",
    "write_sharded",
]

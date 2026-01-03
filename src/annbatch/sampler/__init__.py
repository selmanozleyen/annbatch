"""Sampler classes for efficient chunk-aligned data access from Zarr stores.

This module provides samplers optimized for chunk-based data access patterns:

- :class:`~annbatch.sampler.Sampler`: Abstract base class for all samplers.
- :class:`~annbatch.sampler.SliceSampler`: Chunk-aligned access for full or
  partial dataset iteration. Supports sharding via ``start_index`` and
  ``end_index`` for distributed training.
- :class:`~annbatch.sampler.RangeGroupSampler`: Group-based sampling with
  distributed and multi-worker support. Each batch contains only elements
  from one group. Interleaving is at LoadRequest level.
- :class:`~annbatch.sampler.InterleavedGroupSampler`: True batch-level
  interleaving where each LoadRequest contains batches from multiple groups.
- :class:`~annbatch.sampler.GroupRange`: Defines a named observation range
  for use with group samplers.

"""

from annbatch.sampler._sampler import (
    GroupRange,
    InterleavedGroupSampler,
    RangeGroupSampler,
    Sampler,
    SliceSampler,
)

# Update __module__ so Sphinx can find the re-exported classes
Sampler.__module__ = "annbatch.sampler"
SliceSampler.__module__ = "annbatch.sampler"
RangeGroupSampler.__module__ = "annbatch.sampler"
InterleavedGroupSampler.__module__ = "annbatch.sampler"
GroupRange.__module__ = "annbatch.sampler"

__all__ = [
    "Sampler",
    "SliceSampler",
    "RangeGroupSampler",
    "InterleavedGroupSampler",
    "GroupRange",
]

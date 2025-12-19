"""Sampler classes for efficient chunk-aligned data access from Zarr stores.

This module provides samplers optimized for different data access patterns:

- :class:`SliceSampler`: Chunk-aligned access for full dataset iteration.
- :class:`RangeSampler`: Efficient access for contiguous index ranges.
- :class:`MaskedSampler`: Access for arbitrary (non-contiguous) index subsets.

Memory Considerations
---------------------
For category-based sampling, memory usage depends on data organization:


"""

from annbatch.sampler._sampler import Sampler, SliceSampler

__all__ = [
    "Sampler",
    "SliceSampler",
]

"""Sampler classes for efficient chunk-aligned data access from Zarr stores.

This module provides samplers optimized for different data access patterns:

- :class:`SliceSampler`: Chunk-aligned access for full dataset iteration.
- :class:`RangeSampler`: Efficient access for contiguous index ranges.
- :class:`MaskedSampler`: Access for arbitrary (non-contiguous) index subsets.
- :class:`CategorySampler`: Category-aware sampling with automatic optimization.

Memory Considerations
---------------------
For category-based sampling, memory usage depends on data organization:

- **Sorted data**: When observations are sorted by category, each category forms
  a contiguous range. `CategorySampler` detects this and uses `RangeSampler`,
  storing only (start, stop) per category - O(n_categories) memory.

- **Unsorted data**: Uses `MaskedSampler` which stores index arrays per category -
  O(n_observations) memory. For very large datasets, consider sorting by category
  before saving to reduce memory overhead during sampling.
"""

from annbatch.sampler._sampler import CategorySampler, MaskedSampler, RangeSampler, Sampler, SliceSampler

__all__ = [
    "CategorySampler",
    "MaskedSampler",
    "RangeSampler",
    "Sampler",
    "SliceSampler",
]

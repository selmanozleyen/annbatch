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

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from annbatch.utils import WorkerHandle

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import ArrayLike


class Sampler[T_co](ABC):
    """Base class for all samplers."""

    @abstractmethod
    def __iter__(self) -> Iterator[T_co]:
        """Iterate over batch access patterns."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of batches."""
        ...


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _indices_to_slices(indices: np.ndarray) -> list[slice]:
    """Convert sorted indices to slices for consecutive runs."""
    if len(indices) == 0:
        return []
    breaks = np.where(np.diff(indices) != 1)[0] + 1
    starts = np.concatenate([[0], breaks])
    ends = np.concatenate([breaks, [len(indices)]])
    return [slice(int(indices[s]), int(indices[e - 1]) + 1) for s, e in zip(starts, ends, strict=True)]


def _shuffle_for_worker(arr: np.ndarray) -> np.ndarray:
    """Shuffle array and return this worker's partition."""
    worker = WorkerHandle()
    worker.shuffle(arr)
    return worker.get_part_for_worker(arr)


def _generate_chunk_slices(n_obs: int, chunk_size: int, chunk_indices: np.ndarray) -> list[slice]:
    """Generate slices from chunk indices."""
    starts = chunk_indices * chunk_size
    stops = np.minimum(starts + chunk_size, n_obs)
    return [slice(int(a), int(b)) for a, b in zip(starts, stops, strict=True)]


def _batch_by_obs_count(slices: list[slice], batch_size: int) -> Iterator[list[slice]]:
    """Yield slices grouped into batches of approximately batch_size observations."""
    obs_count = 0
    batch: list[slice] = []

    for s in slices:
        batch.append(s)
        obs_count += s.stop - s.start
        if obs_count >= batch_size:
            yield batch
            batch, obs_count = [], 0

    if batch:
        yield batch


def _is_sorted_categories(categories: np.ndarray) -> bool:
    """Check if categories form contiguous groups (data is sorted by category)."""
    if len(categories) == 0:
        return True
    # Categories are sorted if each unique value appears in one contiguous block
    # This means: when we see a new category, we never see a previous one again
    seen = set()
    prev = categories[0]
    seen.add(prev)
    for cat in categories[1:]:
        if cat != prev:
            if cat in seen:
                return False
            seen.add(cat)
            prev = cat
    return True


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------


class SliceSampler(Sampler[list[slice]]):
    """Chunk-aligned sampler for efficient Zarr store access.

    Parameters
    ----------
    n_obs
        Total number of observations.
    batch_size
        Target total observations per batch.
    chunk_size
        Size of each chunk slice.
    shuffle
        Whether to shuffle chunk order.
    """

    __slots__ = ("_n_obs", "_batch_size", "_chunk_size", "_shuffle", "_n_iters")

    def __init__(self, *, n_obs: int, batch_size: int, chunk_size: int, shuffle: bool = False):
        self._n_obs = n_obs
        self._batch_size = batch_size
        self._chunk_size = chunk_size
        self._shuffle = shuffle
        self._n_iters = math.ceil(n_obs / batch_size)

    def __len__(self) -> int:
        return self._n_iters

    def __iter__(self) -> Iterator[list[slice]]:
        n_chunks = math.ceil(self._n_obs / self._chunk_size)
        chunk_indices = np.arange(n_chunks)

        if self._shuffle:
            chunk_indices = _shuffle_for_worker(chunk_indices)

        slices = _generate_chunk_slices(self._n_obs, self._chunk_size, chunk_indices)
        yield from _batch_by_obs_count(slices, self._batch_size)


class RangeSampler(Sampler[list[slice]]):
    """Sampler for a contiguous index range. Memory efficient: O(1).

    Parameters
    ----------
    start
        Start index (inclusive).
    stop
        Stop index (exclusive).
    batch_size
        Number of observations per batch.
    shuffle
        Whether to shuffle the range before batching.
    """

    __slots__ = ("_start", "_stop", "_batch_size", "_shuffle", "_n_iters")

    def __init__(self, *, start: int, stop: int, batch_size: int, shuffle: bool = False):
        self._start = start
        self._stop = stop
        self._batch_size = batch_size
        self._shuffle = shuffle
        n_obs = stop - start
        self._n_iters = math.ceil(n_obs / batch_size) if n_obs > 0 else 0

    def __len__(self) -> int:
        return self._n_iters

    def __iter__(self) -> Iterator[list[slice]]:
        n_obs = self._stop - self._start
        if n_obs == 0:
            return

        if self._shuffle:
            # Need to materialize indices for shuffling
            indices = _shuffle_for_worker(np.arange(self._start, self._stop))
            for i in range(0, len(indices), self._batch_size):
                batch_indices = np.sort(indices[i : i + self._batch_size])
                yield _indices_to_slices(batch_indices)
        else:
            # Efficient: yield contiguous slices without materializing indices
            for i in range(0, n_obs, self._batch_size):
                yield [slice(self._start + i, min(self._start + i + self._batch_size, self._stop))]


class MaskedSampler(Sampler[list[slice]]):
    """Sampler for arbitrary index subsets. Memory: O(n_indices).

    Parameters
    ----------
    indices
        Array of observation indices to sample from.
    batch_size
        Number of indices per batch.
    shuffle
        Whether to shuffle indices.
    """

    __slots__ = ("_indices", "_batch_size", "_shuffle", "_n_iters")

    def __init__(self, *, indices: ArrayLike, batch_size: int, shuffle: bool = False):
        self._indices = np.asarray(indices)
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._n_iters = math.ceil(len(self._indices) / batch_size) if len(self._indices) else 0

    def __len__(self) -> int:
        return self._n_iters

    def __iter__(self) -> Iterator[list[slice]]:
        if len(self._indices) == 0:
            return

        indices = _shuffle_for_worker(self._indices.copy()) if self._shuffle else self._indices

        for i in range(0, len(indices), self._batch_size):
            batch_indices = np.sort(indices[i : i + self._batch_size])
            yield _indices_to_slices(batch_indices)


class CategorySampler(Sampler[list[slice]]):
    """Category-aware sampler with automatic memory optimization.

    Iterates through categories, yielding batched slices for each. Automatically
    detects whether data is sorted by category:

    - **Sorted**: Uses `RangeSampler` per category - O(n_categories) memory.
    - **Unsorted**: Uses `MaskedSampler` per category - O(n_observations) memory.

    Parameters
    ----------
    categories
        Array of category labels for each observation.
    batch_size
        Number of observations per batch within each category.
    shuffle_categories
        Whether to shuffle category iteration order.
    shuffle_within
        Whether to shuffle observations within each category.

    Notes
    -----
    For large datasets, sorting data by category before saving significantly
    reduces memory usage during sampling.
    """

    __slots__ = (
        "_batch_size",
        "_shuffle_categories",
        "_shuffle_within",
        "_unique_categories",
        "_category_data",
        "_n_iters",
        "_is_sorted",
    )

    def __init__(
        self,
        *,
        categories: ArrayLike,
        batch_size: int,
        shuffle_categories: bool = False,
        shuffle_within: bool = False,
    ):
        categories_arr = np.asarray(categories)
        self._batch_size = batch_size
        self._shuffle_categories = shuffle_categories
        self._shuffle_within = shuffle_within
        self._is_sorted = _is_sorted_categories(categories_arr)

        self._unique_categories, first_idx, counts = np.unique(categories_arr, return_index=True, return_counts=True)

        if self._is_sorted:
            # Store (start, stop) ranges - O(n_categories) memory
            self._category_data = {
                cat: (int(start), int(start + count))
                for cat, start, count in zip(self._unique_categories, first_idx, counts, strict=True)
            }
        else:
            # Store index arrays - O(n_observations) memory
            self._category_data = {cat: np.flatnonzero(categories_arr == cat) for cat in self._unique_categories}

        self._n_iters = sum(math.ceil(c / batch_size) for c in counts)

    def __len__(self) -> int:
        return self._n_iters

    def __iter__(self) -> Iterator[list[slice]]:
        categories = self._unique_categories.copy()

        if self._shuffle_categories:
            WorkerHandle().shuffle(categories)

        for cat in categories:
            data = self._category_data[cat]
            if self._is_sorted:
                start, stop = data
                yield from RangeSampler(
                    start=start, stop=stop, batch_size=self._batch_size, shuffle=self._shuffle_within
                )
            else:
                yield from MaskedSampler(indices=data, batch_size=self._batch_size, shuffle=self._shuffle_within)

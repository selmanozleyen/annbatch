"""Sampler classes for efficient chunk-aligned data access from Zarr stores."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from annbatch.sampler._utils import (
    batch_by_obs_count,
    generate_chunk_slices,
    indices_to_slices,
    is_sorted_categories,
    shuffle_for_worker,
)
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
            chunk_indices = shuffle_for_worker(chunk_indices)

        slices = generate_chunk_slices(self._n_obs, self._chunk_size, chunk_indices)
        yield from batch_by_obs_count(slices, self._batch_size)


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
            indices = shuffle_for_worker(np.arange(self._start, self._stop))
            for i in range(0, len(indices), self._batch_size):
                batch_indices = np.sort(indices[i : i + self._batch_size])
                yield indices_to_slices(batch_indices)
        else:
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

        indices = shuffle_for_worker(self._indices.copy()) if self._shuffle else self._indices

        for i in range(0, len(indices), self._batch_size):
            batch_indices = np.sort(indices[i : i + self._batch_size])
            yield indices_to_slices(batch_indices)


class CategorySampler(Sampler[list[slice]]):
    """Category-aware sampler with automatic memory optimization.

    Iterates through categories, yielding batched slices for each.

    - **Sorted**: Uses `RangeSampler` per category - O(n_categories) memory.
    - **Unsorted**: Uses `MaskedSampler` per category - O(n_observations) memory.

    Parameters
    ----------
    categories
        Array of category labels for each observation.
    batch_size
        Number of observations per batch within each category.
    sorted_categories
        Whether categories form contiguous groups (data sorted by category).
        If None (default), auto-detects. Set explicitly to skip detection.
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
        sorted_categories: bool | None = None,
        shuffle_categories: bool = False,
        shuffle_within: bool = False,
    ):
        categories_arr = np.asarray(categories)
        self._batch_size = batch_size
        self._shuffle_categories = shuffle_categories
        self._shuffle_within = shuffle_within
        self._is_sorted = is_sorted_categories(categories_arr) if sorted_categories is None else sorted_categories

        self._unique_categories, first_idx, counts = np.unique(categories_arr, return_index=True, return_counts=True)

        if self._is_sorted:
            self._category_data = {
                cat: (int(start), int(start + count))
                for cat, start, count in zip(self._unique_categories, first_idx, counts, strict=True)
            }
        else:
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

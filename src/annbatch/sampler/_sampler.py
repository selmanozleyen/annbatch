"""Sampler classes for efficient chunk-aligned data access from Zarr stores."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


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

    def __init__(self, *, n_obs: int, batch_size: int, chunk_size: int, shuffle: bool = False, preload_nchunks: int):
        if preload_nchunks < 1:
            raise ValueError("preload_nchunks must be >= 1")
        self._n_obs = n_obs
        self._batch_size = batch_size
        self._chunk_size = chunk_size
        self._shuffle = shuffle
        self._n_iters = math.ceil(n_obs / (self._chunk_size * preload_nchunks))

    def __len__(self) -> int:
        return self._n_iters

    def __iter__(self) -> Iterator[list[slice]]:
        pass
        # would have
        # for s in sampler:
        #     assert sum( s.stop - s.start for s in s) == self._chunk_size * preload_nchunks


class CategorySampler(SliceSampler):

    def __iter__(self) -> Iterator[list[slice]]:
        # each iteration should yield a list of slices that sum to chunk_size * preload_nchunks
        # the sum of the slice ranges within one category should be


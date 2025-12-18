"""Sampler classes for efficient chunk-aligned data access from Zarr stores."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from annbatch.sampler._utils import batch_by_obs_count, generate_chunk_slices, shuffle_for_worker

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

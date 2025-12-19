"""Sampler classes for efficient chunk-aligned data access from Zarr stores."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from itertools import islice
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Iterator

    from annbatch.utils import WorkerHandle


def _batched[T](iterable: Iterable[T], n: int) -> Generator[list[T], None, None]:
    if n < 1:
        raise ValueError("n must be >= 1")
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


class Sampler[T_co](ABC):
    """Base class for all samplers."""

    @abstractmethod
    def __iter__(self) -> Iterator[T_co, list[np.ndarray]]:
        """Iterate over batch access patterns."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of iterations to exhaust the sampler."""
        pass


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

    __slots__ = ("_n_obs", "_batch_size", "_chunk_size", "_shuffle", "_n_iters", "_worker_handle")

    def __init__(
        self,
        *,
        n_obs: int,
        batch_size: int,
        chunk_size: int,
        shuffle: bool = False,
        preload_nchunks: int,
        worker_handle: WorkerHandle | None = None,
    ):
        if preload_nchunks < 1:
            raise ValueError("preload_nchunks must be >= 1")
        self._n_obs = n_obs
        self._batch_size = batch_size
        self._chunk_size = chunk_size
        self._shuffle = shuffle
        self._preload_nchunks = preload_nchunks
        self._n_iters = math.ceil(n_obs / (self._chunk_size * preload_nchunks))
        self._n_chunks = math.ceil(n_obs / chunk_size)
        self._worker_handle = worker_handle

    def __len__(self) -> int:
        return self._n_iters

    def _shuffle_integers(self, integers: np.ndarray) -> np.ndarray:
        if self._worker_handle is None:
            # todo: deal with generators later
            np.random.default_rng().shuffle(integers)
        else:
            # should we always have a worker handle? even if its no-op?
            self._worker_handle.shuffle(integers)
        return integers

    def _chunk_ids_to_slices(self, chunk_ids: np.ndarray) -> list[slice]:
        return [
            slice(chunk_id * self._chunk_size, min((chunk_id + 1) * self._chunk_size, self._n_obs))
            for chunk_id in chunk_ids
        ]

    def __iter__(self) -> Iterator[tuple[list[slice], list[np.ndarray]]]:
        chunk_ids = np.arange(self._n_chunks)
        # indices applied to the loaded data
        preloaded_indices = np.arange(self._preload_nchunks * self._chunk_size)
        if self._shuffle:
            preloaded_indices = self._shuffle_integers(preloaded_indices)
            chunk_ids = self._shuffle_integers(chunk_ids)
        splits = np.split(preloaded_indices, np.arange(self._batch_size, len(preloaded_indices), self._batch_size))

        for preloaded_chunk_ids in _batched(chunk_ids, self._preload_nchunks):
            yield (
                self._chunk_ids_to_slices(preloaded_chunk_ids),
                splits,
            )

"""Sampler classes for efficient chunk-aligned data access from Zarr stores."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

    from annbatch.utils import WorkerHandle


class Sampler[T_co](ABC):
    """Base sampler class."""

    @abstractmethod
    def __iter__(self) -> Iterator[tuple[T_co, list[np.ndarray], np.ndarray | None]]:
        """Yield (slices, batch_indices, leftover_indices) tuples."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of iterations to exhaust the sampler."""


class SliceSampler(Sampler[list[slice]]):
    """Chunk-based slice sampler for batched data access.

    Parameters
    ----------
    n_obs
        Total number of observations in the dataset.
    batch_size
        Number of observations per batch.
    chunk_size
        Size of each chunk in the backing store.
    shuffle
        Whether to shuffle chunk and index order.
    preload_nchunks
        Number of chunks to load per iteration.
    drop_last
        Whether to drop the last incomplete batch.
    worker_handle
        Optional handle for worker-specific shuffling.
    start_index
        Starting observation index (inclusive).
    end_index
        Ending observation index (exclusive). Defaults to `n_obs`.
    rng
        Random number generator for shuffling.
    """

    __slots__ = (
        "_n_obs",
        "_batch_size",
        "_chunk_size",
        "_shuffle",
        "_preload_nchunks",
        "_start_index",
        "_end_index",
        "_start_chunk_id",
        "_end_chunk_id",
        "_n_chunks_to_load",
        "_n_chunk_iters",
        "_n_iters",
        "_worker_handle",
        "_drop_last",
        "_rng",
    )

    def __init__(
        self,
        *,
        n_obs: int,
        batch_size: int,
        chunk_size: int,
        shuffle: bool = False,
        preload_nchunks: int,
        drop_last: bool = False,
        worker_handle: WorkerHandle | None = None,
        start_index: int = 0,
        end_index: int | None = None,
        rng: np.random.Generator | None = None,
    ):
        if preload_nchunks < 1:
            raise ValueError("preload_nchunks must be >= 1")
        self._n_obs = n_obs
        self._batch_size = batch_size
        self._chunk_size = chunk_size
        self._shuffle = shuffle
        self._preload_nchunks = preload_nchunks

        if start_index < 0 or start_index >= n_obs:
            raise ValueError("start_index must be >= 0 and < n_obs")
        self._start_index = start_index
        if end_index is not None:
            if end_index < 0 or end_index > n_obs or end_index < start_index:
                raise ValueError("end_index must be >= 0 and < n_obs and > start_index")
            self._end_index = end_index
        else:
            self._end_index = n_obs

        self._start_chunk_id = self._start_index // self._chunk_size
        self._end_chunk_id = (self._end_index - 1) // self._chunk_size

        self._n_chunks_to_load = self._end_chunk_id - self._start_chunk_id + 1
        self._n_chunk_iters = math.ceil(self._n_chunks_to_load / preload_nchunks)

        n_batches = (
            math.floor((self._end_index - self._start_index) / self._batch_size)
            if drop_last
            else math.ceil((self._end_index - self._start_index) / self._batch_size)
        )
        total_yielded_obs = n_batches * self._batch_size
        self._n_iters = math.ceil(total_yielded_obs / (self._chunk_size * preload_nchunks))
        self._worker_handle = worker_handle
        self._drop_last = drop_last

        self._rng = np.random.default_rng() if rng is None else rng

    def __len__(self) -> int:
        return self._n_iters

    def __iter__(self) -> Iterator[tuple[list[slice], list[np.ndarray], np.ndarray | None]]:
        chunk_ids = np.arange(self._n_chunks_to_load) + self._start_chunk_id

        # precompute slices before shuffling
        slices = self._chunk_ids_to_slices(chunk_ids)
        if self._n_chunks_to_load == 1:
            slices[0] = slice(
                self._start_index,
                self._end_index,
            )
        else:
            slices[0] = slice(
                self._start_index,
                min((self._start_chunk_id + 1) * self._chunk_size, self._end_index),
            )
            slices[-1] = slice(
                self._end_chunk_id * self._chunk_size,
                self._end_index,
            )

        if self._shuffle:
            chunk_ids = self._shuffle_integers(chunk_ids)

        assert len(chunk_ids) == self._n_chunks_to_load

        n_leftover_loaded_indices = 0
        leftover_split = None

        for i in range(self._n_chunk_iters):
            start = i * self._preload_nchunks
            end = min(start + self._preload_nchunks, len(chunk_ids))
            chunk_ids_to_load = chunk_ids[start:end] + self._start_chunk_id
            # for smaller preload_nchunks, below is not expensive but for large preload_nchunks,
            # maybe it would be worth consideration to precompute the slice_sizes and use them here
            total_obs_to_load = sum(slices[i].stop - slices[i].start for i in chunk_ids_to_load)
            # splits is a list of arrays of indices that will be used to index the data after it is loaded
            loaded_indices = np.arange(total_obs_to_load + n_leftover_loaded_indices)
            if self._shuffle:
                loaded_indices = self._shuffle_integers(loaded_indices)
            splits = np.split(loaded_indices, np.arange(self._batch_size, len(loaded_indices), self._batch_size))

            # if the last batch is full, there is no leftover data
            if splits[-1].shape[0] == self._batch_size:
                n_leftover_loaded_indices = 0
                leftover_split = None
            # handle leftover data
            else:
                is_last_iter = i == self._n_chunk_iters - 1

                if is_last_iter and not self._drop_last:
                    # Yield the final partial batch
                    n_leftover_loaded_indices = 0
                    leftover_split = None
                else:
                    # Either save for next iteration, or drop on last iter
                    n_leftover_loaded_indices = splits[-1].shape[0]
                    leftover_split = None if (self._drop_last and is_last_iter) else splits[-1]
                    splits = splits[:-1]

            yield (
                [slices[i - self._start_chunk_id] for i in chunk_ids_to_load],
                splits,
                leftover_split,
            )

    def _shuffle_integers(self, integers: np.ndarray) -> np.ndarray:
        if self._worker_handle is None:
            # TODO: deal with generators later
            self._rng.shuffle(integers)
        else:
            # should we always have a worker handle? even if its no-op?
            self._worker_handle.shuffle(integers)
        return integers

    def _chunk_ids_to_slices(self, chunk_ids: np.ndarray) -> list[slice]:
        return [
            slice(chunk_id * self._chunk_size, min((chunk_id + 1) * self._chunk_size, self._n_obs))
            for chunk_id in chunk_ids
        ]

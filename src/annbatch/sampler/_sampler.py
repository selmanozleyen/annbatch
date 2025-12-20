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
    def __iter__(self) -> Iterator[tuple[T_co, list[np.ndarray], np.ndarray | None]]:
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
        "_n_chunks_total",
        "_n_chunk_iters",
        "_n_batches",
        "_total_yielded_obs",
        "_n_iters",
        "_worker_handle",
        "_drop_last",
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
        self._n_chunks_total = math.ceil(self._n_obs / self._chunk_size)
        self._n_chunk_iters = math.ceil(self._n_chunks_to_load / preload_nchunks)

        self._n_batches = (
            math.floor((self._end_index - self._start_index) / self._batch_size)
            if drop_last
            else math.ceil((self._end_index - self._start_index) / self._batch_size)
        )
        # this is the total number of observations that will be yielded
        # and won't be ignored when applying the batch indices.
        self._total_yielded_obs = self._n_batches * self._batch_size
        self._n_iters = math.ceil(self._total_yielded_obs / (self._chunk_size * preload_nchunks))
        self._worker_handle = worker_handle
        self._drop_last = drop_last

    def __len__(self) -> int:
        return self._n_iters

    def _shuffle_integers(self, integers: np.ndarray) -> np.ndarray:
        if self._worker_handle is None:
            # TODO: deal with generators later
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

    def __iter__(self) -> Iterator[tuple[list[slice], list[np.ndarray], np.ndarray | None]]:
        chunk_ids = np.arange(self._n_chunks_to_load) + self._start_chunk_id
        # indices applied to the loaded data

        # # data needs to be preserved from previous batches if below is not 0
        # n_leftover_each_batch = (self._preload_nchunks * self._chunk_size) % self._batch_size
        # # last batch data will be ommited if below is not 0 and drop_last is True
        # n_leftover_total = self._n_obs % self._batch_size

        # is this memory expensive?
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

        for i, chunk_ids_to_load in enumerate(_batched(chunk_ids, self._preload_nchunks)):
            # maybe below can be avoided
            total_obs_to_load = sum(
                slices[i - self._start_chunk_id].stop - slices[i - self._start_chunk_id].start
                for i in chunk_ids_to_load
            )
            # splits is a list of arrays of indices that will be used to index the data after it is loaded
            loaded_indices = np.arange(total_obs_to_load + n_leftover_loaded_indices)
            if self._shuffle:
                loaded_indices = self._shuffle_integers(loaded_indices)
            splits = np.split(loaded_indices, np.arange(self._batch_size, len(loaded_indices), self._batch_size))

            if splits[-1].shape[0] == self._batch_size:
                n_leftover_loaded_indices = 0
                leftover_split = None
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

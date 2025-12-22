"""Sampler classes for efficient chunk-aligned data access from Zarr stores."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from annbatch.utils import check_lt_1

if TYPE_CHECKING:
    from collections.abc import Iterator

    from annbatch.utils import WorkerHandle


T_co = TypeVar("T_co", covariant=True)


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
        "_n_chunks",
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
        check_lt_1([chunk_size, preload_nchunks], ["Chunk size", "Preload chunks"])

        if batch_size > (chunk_size * preload_nchunks):
            raise ValueError(
                "batch_size cannot exceed chunk_size * preload_nchunks. "
                f"Got batch_size={batch_size}, but max is {chunk_size * preload_nchunks}."
            )

        # Worker mode validation
        if worker_handle is not None:
            preload_size = chunk_size * preload_nchunks
            if not drop_last and preload_size % batch_size != 0:
                raise ValueError(
                    f"When using DataLoader workers with drop_last=False, "
                    f"(chunk_size * preload_nchunks) must be divisible by batch_size. "
                    f"Got {preload_size} % {batch_size} = {preload_size % batch_size}. "
                    f"Set drop_last=True to allow non-divisible configs."
                )
            if drop_last:
                import warnings

                warnings.warn(
                    f"With drop_last=True and multiple workers, up to "
                    f"(batch_size - 1) * num_workers observations may be dropped "
                    f"(one partial batch per worker).",
                    UserWarning,
                    stacklevel=2,
                )

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

        # Compute number of chunks spanning [start_index, end_index)
        self._n_chunks = math.ceil((self._end_index - self._start_index) / self._chunk_size)

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

    def _compute_slices(self) -> list[slice]:
        """Compute slices directly from start/end indices."""
        starts = list(range(self._start_index, self._end_index, self._chunk_size))
        stops = starts[1:] + [self._end_index]
        return [slice(start, stop) for start, stop in zip(starts, stops)]

    def __iter__(self) -> Iterator[tuple[list[slice], list[np.ndarray], np.ndarray | None]]:
        # Compute slices directly from index range
        slices = self._compute_slices()
        n_slices = len(slices)

        # Create slice indices for shuffling
        slice_indices = np.arange(n_slices)
        if self._shuffle:
            slice_indices = self._shuffle_integers(slice_indices)

        # Worker sharding: each worker gets a disjoint subset of slices
        if self._worker_handle is not None:
            slice_indices = self._worker_handle.get_part_for_worker(slice_indices)

        n_slices_for_worker = len(slice_indices)
        n_slice_iters = math.ceil(n_slices_for_worker / self._preload_nchunks) if n_slices_for_worker > 0 else 0

        n_leftover_loaded_indices = 0
        leftover_split = None

        for i in range(n_slice_iters):
            start = i * self._preload_nchunks
            end = min(start + self._preload_nchunks, n_slices_for_worker)
            indices_to_load = slice_indices[start:end]

            # Compute total observations to load from selected slices
            total_obs_to_load = sum(slices[idx].stop - slices[idx].start for idx in indices_to_load)

            # Generate loaded indices with leftover from previous iteration
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
                is_last_iter = i == n_slice_iters - 1

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
                [slices[idx] for idx in indices_to_load],
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

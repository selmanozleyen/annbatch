"""Sampler classes for efficient chunk-based data access."""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING

import numpy as np

from annbatch.abc import Sampler
from annbatch.samplers._utils import get_torch_worker_info
from annbatch.utils import _spawn_worker_rng, check_lt_1, split_given_size

if TYPE_CHECKING:
    from collections.abc import Iterator

    from annbatch.samplers._utils import WorkerInfo
    from annbatch.types import LoadRequest


class ChunkSampler(Sampler):
    """Chunk-based sampler for batched data access.

    Parameters
    ----------
    batch_size
        Number of observations per batch.
    chunk_size
        Size of each chunk i.e. the range of each chunk yielded.
    mask
        A slice defining the observation range to sample from (start:stop).
    shuffle
        Whether to shuffle chunk and index order.
    preload_nchunks
        Number of chunks to load per iteration.
    drop_last
        Whether to drop the last incomplete batch.
    rng
        Random number generator for shuffling. Note that :func:`torch.manual_seed`
        has no effect on reproducibility here; pass a seeded
        :class:`numpy.random.Generator` to control randomness.
    """

    _batch_size: int
    _chunk_size: int
    _shuffle: bool
    _preload_nchunks: int
    _mask: slice
    _drop_last: bool
    _rng: np.random.Generator

    def __init__(
        self,
        chunk_size: int,
        preload_nchunks: int,
        batch_size: int,
        *,
        mask: slice | None = None,
        shuffle: bool = False,
        drop_last: bool = False,
        rng: np.random.Generator | None = None,
    ):
        self._mask = _get_mask_after_validation(mask)
        self._in_memory_size = _get_preload_size_after_validation(chunk_size, preload_nchunks, batch_size)
        self._rng = rng or np.random.default_rng()
        self._batch_size, self._chunk_size, self._shuffle = batch_size, chunk_size, shuffle
        self._preload_nchunks, self._drop_last = preload_nchunks, drop_last

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def shuffle(self) -> bool:
        return self._shuffle

    def n_iters(self, n_obs: int) -> int:
        start, stop = _resolve_start_stop(self._mask, n_obs)
        total_obs = stop - start
        return total_obs // self._batch_size if self._drop_last else math.ceil(total_obs / self._batch_size)

    def validate(self, n_obs: int) -> None:
        """Validate the sampler configuration against the loader's n_obs.

        Parameters
        ----------
        n_obs
            The total number of observations in the loader.

        Raises
        ------
        ValueError
            If the sampler configuration is invalid for the given n_obs.
        """
        _validate_start_stop(self._mask, n_obs)

    def _validate_worker_mode(self, worker_info: WorkerInfo | None) -> None:
        if worker_info is not None and worker_info.num_workers > 1 and not self._drop_last and self._batch_size != 1:
            raise ValueError("When using DataLoader with multiple workers drop_last=False is not supported.")

    def _sample(self, n_obs: int) -> Iterator[LoadRequest]:
        worker_info = get_torch_worker_info()
        self._validate_worker_mode(worker_info)

        start, stop = _resolve_start_stop(self._mask, n_obs)
        worker_aware_rng = self._rng if worker_info is None else _spawn_worker_rng(self._rng, worker_info.id)

        chunks = _compute_chunks_sequential(start, stop, self._chunk_size, self._shuffle, rng=self._rng)
        yield from _iter_load_requests(
            chunks,
            batch_size=self._batch_size,
            in_memory_size=self._in_memory_size,
            preload_nchunks=self._preload_nchunks,
            shuffle=self._shuffle,
            drop_last=self._drop_last,
            batch_rng=worker_aware_rng,
            worker_info=worker_info,
        )


class ChunkSamplerWithReplacement(Sampler):
    """Chunk-based sampler that draws chunks with replacement.

    Unlike :class:`ChunkSampler`, this sampler draws random contiguous
    chunks from the observation range with replacement and is not limited
    to a single epoch. The number of batches to yield (``n_iters``) is required.

    See :class:`ChunkSampler` for the shared parameters.

    Parameters
    ----------
    n_iters
        Number of batches to yield.
    """

    def __init__(
        self,
        chunk_size: int,
        preload_nchunks: int,
        batch_size: int,
        *,
        n_iters: int,
        mask: slice | None = None,
        rng: np.random.Generator | None = None,
    ):
        check_lt_1([n_iters], ["n_iters"])
        self._mask = _get_mask_after_validation(mask)
        self._in_memory_size = _get_preload_size_after_validation(chunk_size, preload_nchunks, batch_size)
        self._rng = rng or np.random.default_rng()
        self._batch_size, self._chunk_size = batch_size, chunk_size
        self._preload_nchunks, self._n_iters = preload_nchunks, n_iters

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def shuffle(self) -> bool:
        return True

    def n_iters(self, n_obs: int) -> int:
        return self._n_iters

    def validate(self, n_obs: int) -> None:
        """Validate the sampler configuration against the loader's n_obs.

        Parameters
        ----------
        n_obs
            The total number of observations in the loader.

        Raises
        ------
        ValueError
            If the sampler configuration is invalid for the given n_obs.
        """
        _validate_start_stop(self._mask, n_obs)

    def _sample(self, n_obs: int) -> Iterator[LoadRequest]:
        worker_info = get_torch_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            raise ValueError("Multiple workers are not supported with this sampler.")

        start, stop = _resolve_start_stop(self._mask, n_obs)
        n_total_chunks = math.ceil((self._n_iters * self._batch_size) / self._chunk_size)
        chunks = _compute_chunks_random(start, stop, self._chunk_size, n_total_chunks, rng=self._rng)

        load_requests = _iter_load_requests(
            chunks,
            batch_size=self._batch_size,
            in_memory_size=self._in_memory_size,
            preload_nchunks=self._preload_nchunks,
            shuffle=True,
            drop_last=False,
            batch_rng=self._rng,
            worker_info=None,
        )
        batches_per_request = self._in_memory_size // self._batch_size
        chunks_per_batch = self._batch_size // self._chunk_size
        n_full, tail = divmod(self._n_iters, batches_per_request)
        yield from itertools.islice(load_requests, n_full)
        if tail > 0:
            load_request = next(load_requests)
            yield {
                "chunks": load_request["chunks"][:chunks_per_batch],
                "splits": load_request["splits"][:tail],
            }


def _get_mask_after_validation(mask: slice | None) -> slice:
    """Validate and normalize a mask slice, returning ``slice(start, stop)``."""
    if mask is None:
        mask = slice(0, None)
    if mask.step is not None and mask.step != 1:
        raise ValueError(f"mask.step must be 1, but got {mask.step}")
    start, stop = mask.start or 0, mask.stop
    if start < 0:
        raise ValueError("mask.start must be >= 0")
    if stop is not None and start >= stop:
        raise ValueError("mask.start must be < mask.stop when mask.stop is specified")
    return slice(start, stop)


def _resolve_start_stop(mask: slice, n_obs: int) -> tuple[int, int]:
    return mask.start or 0, mask.stop or n_obs


def _get_preload_size_after_validation(chunk_size: int, preload_nchunks: int, batch_size: int) -> int:
    """Compute preload size after validation."""
    check_lt_1([chunk_size, preload_nchunks], ["Chunk size", "Preloaded chunks"])
    preload_size = chunk_size * preload_nchunks
    if batch_size > preload_size:
        raise ValueError(
            "batch_size cannot exceed chunk_size * preload_nchunks. "
            f"Got batch_size={batch_size}, but max is {preload_size}."
        )
    if preload_size % batch_size != 0:
        raise ValueError(
            "chunk_size * preload_nchunks must be divisible by batch_size. "
            f"Got {preload_size} % {batch_size} = {preload_size % batch_size}."
        )
    return preload_size


def _validate_start_stop(mask: slice, n_obs: int) -> None:
    """Validate that the resolved mask range is valid for the given n_obs."""
    start, stop = _resolve_start_stop(mask, n_obs)
    if stop > n_obs:
        raise ValueError(
            f"Sampler mask.stop ({stop}) exceeds loader n_obs ({n_obs}). "
            "The sampler range must be within the loader's observations."
        )
    if start >= stop:
        raise ValueError(f"Sampler mask.start ({start}) must be < mask.stop ({stop}).")


def _compute_chunks_sequential(
    start: int, stop: int, chunk_size: int, shuffle: bool, rng: np.random.Generator
) -> list[slice]:
    """Compute chunks from start and stop indices.

    The chunks are computed such that the last chunk is the incomplete chunk
    if the total number of observations is not divisible by the chunk size.
    Also works with shuffled chunk indices so that the incomplete chunk is
    not always in the same position.
    """
    chunk_indices = np.arange(math.ceil((stop - start) / chunk_size))
    if shuffle:
        rng.shuffle(chunk_indices)
    n_chunks, pivot_index = len(chunk_indices), chunk_indices[-1]
    offsets = np.ones(n_chunks + 1, dtype=int) * chunk_size
    offsets[0] = start
    offsets[pivot_index + 1] = incomplete if (incomplete := (stop - start) % chunk_size) else chunk_size
    offsets = np.cumsum(offsets)
    starts, stops = offsets[:-1][chunk_indices], offsets[1:][chunk_indices]
    return [slice(int(s), int(e)) for s, e in zip(starts, stops, strict=True)]


def _compute_chunks_random(
    start: int, stop: int, chunk_size: int, n_chunks: int, rng: np.random.Generator
) -> list[slice]:
    """Draw random contiguous chunks with replacement from the observation range."""
    if stop - start < chunk_size:
        return [slice(start, stop)]
    start_indices = rng.integers(start, stop - chunk_size + 1, size=n_chunks)
    return [slice(int(s), int(s + chunk_size)) for s in start_indices]


def _iter_load_requests(
    chunks: list[slice],
    batch_size: int,
    in_memory_size: int,
    preload_nchunks: int,
    shuffle: bool,
    drop_last: bool,
    batch_rng: np.random.Generator,
    worker_info: WorkerInfo | None,
) -> Iterator[LoadRequest]:
    """Iterate over chunks and yield LoadRequests with splits."""
    if worker_info is not None:
        chunks = np.array_split(chunks, worker_info.num_workers)[worker_info.id]
    chunks_per_request = split_given_size(chunks, preload_nchunks)
    batch_indices = np.arange(in_memory_size)
    split_batch_indices = split_given_size(batch_indices, batch_size)
    for request_chunks in chunks_per_request[:-1]:
        if shuffle:
            batch_rng.shuffle(batch_indices)
            split_batch_indices = split_given_size(batch_indices, batch_size)
        yield {"chunks": request_chunks, "splits": split_batch_indices}
    final_chunks = chunks_per_request[-1]
    total_obs_in_last_batch = int(sum(s.stop - s.start for s in final_chunks))
    if total_obs_in_last_batch == 0:  # pragma: no cover
        raise RuntimeError("Last batch was found to have no observations. Please open an issue.")
    if drop_last:
        if total_obs_in_last_batch < batch_size:
            return
        total_obs_in_last_batch -= total_obs_in_last_batch % batch_size
    indices = batch_rng.permutation(total_obs_in_last_batch) if shuffle else np.arange(total_obs_in_last_batch)
    batch_indices = split_given_size(indices, batch_size)
    yield {"chunks": final_chunks, "splits": batch_indices}

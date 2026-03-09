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

type _ChunkInfo = tuple[np.ndarray, np.ndarray]


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
    _in_memory_size: int

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
        if mask is None:
            mask = slice(0, None)
        if mask.step is not None and mask.step != 1:
            raise ValueError(f"mask.step must be 1, but got {mask.step}")
        start, stop = mask.start or 0, mask.stop
        if start < 0:
            raise ValueError("mask.start must be >= 0")
        if stop is not None and start >= stop:
            raise ValueError("mask.start must be < mask.stop when mask.stop is specified")

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
        self._rng = rng or np.random.default_rng()
        self._in_memory_size = chunk_size * preload_nchunks
        self._batch_size, self._chunk_size, self._shuffle = batch_size, chunk_size, shuffle
        self._preload_nchunks, self._mask, self._drop_last = (
            preload_nchunks,
            slice(start, stop),
            drop_last,
        )

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def shuffle(self) -> bool:
        return self._shuffle

    def n_iters(self, n_obs: int) -> int:
        start, stop = self._resolve_start_stop(n_obs)
        total_obs = stop - start
        return total_obs // self.batch_size if self._drop_last else math.ceil(total_obs / self.batch_size)

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
        start, stop = self._resolve_start_stop(n_obs)
        if stop > n_obs:
            raise ValueError(
                f"Sampler mask.stop ({stop}) exceeds loader n_obs ({n_obs}). "
                "The sampler range must be within the loader's observations."
            )
        if start >= stop:
            raise ValueError(f"Sampler mask.start ({start}) must be < mask.stop ({stop}).")

    def _validate_worker_mode(self, worker_info: WorkerInfo | None) -> None:
        # Worker mode validation - only check when there are multiple workers
        if worker_info is not None and worker_info.num_workers > 1 and not self._drop_last and self.batch_size != 1:
            # With batch_size=1, every batch is exactly 1 item, so no partial batches exist.
            raise ValueError("When using DataLoader with multiple workers drop_last=False is not supported.")

    def _sample(self, n_obs: int) -> Iterator[LoadRequest]:
        worker_info = get_torch_worker_info()
        self._validate_worker_mode(worker_info)

        start, stop = self._resolve_start_stop(n_obs)
        worker_aware_rng = self._rng if worker_info is None else _spawn_worker_rng(self._rng, worker_info.id)

        chunk_info = self._compute_chunks(start, stop, rng=self._rng)
        yield from self._iter_from_chunks(chunk_info, batch_rng=worker_aware_rng, worker_info=worker_info)

    def _iter_from_chunks(
        self,
        chunk_info: _ChunkInfo,
        batch_rng: np.random.Generator,
        worker_info: WorkerInfo | None,
    ) -> Iterator[LoadRequest]:
        starts, stops = chunk_info
        # Worker sharding: each worker gets a disjoint subset of chunks
        if worker_info is not None:
            parts = np.array_split(np.arange(len(starts)), worker_info.num_workers)
            worker_idx = parts[worker_info.id]
            starts = starts[worker_idx]
            stops = stops[worker_idx]

        n_chunks = len(starts)
        if n_chunks == 0:
            return
        # Split into groups of preload_nchunks
        starts_per_request = split_given_size(starts, self._preload_nchunks)
        stops_per_request = split_given_size(stops, self._preload_nchunks)

        batch_indices = np.arange(self._in_memory_size)
        split_batch_indices = split_given_size(batch_indices, self.batch_size)
        for req_starts, req_stops in zip(starts_per_request[:-1], stops_per_request[:-1], strict=True):
            if self.shuffle:
                batch_rng.shuffle(batch_indices)
                split_batch_indices = split_given_size(batch_indices, self.batch_size)
            yield {
                "starts": req_starts,
                "stops": req_stops,
                "splits": split_batch_indices,
            }
        # Last request: may have fewer chunks and/or an incomplete last chunk
        final_starts = starts_per_request[-1]
        final_stops = stops_per_request[-1]
        total_obs_in_last_batch = int((final_stops - final_starts).sum())
        if total_obs_in_last_batch == 0:  # pragma: no cover
            raise RuntimeError("Last batch was found to have no observations. Please open an issue.")
        if self._drop_last:
            if total_obs_in_last_batch < self.batch_size:
                return
            total_obs_in_last_batch -= total_obs_in_last_batch % self.batch_size
        indices = batch_rng.permutation(total_obs_in_last_batch) if self.shuffle else np.arange(total_obs_in_last_batch)
        batch_indices = split_given_size(indices, self.batch_size)
        yield {
            "starts": final_starts,
            "stops": final_stops,
            "splits": batch_indices,
        }

    def _compute_chunks(self, start: int, stop: int, rng: np.random.Generator) -> _ChunkInfo:
        """Compute chunk (start, stop) pairs.

        The last chunk may be smaller than chunk_size when total observations
        is not divisible by chunk_size. Shuffling reorders chunk indices so the
        incomplete chunk is not always at the physical end.
        """
        chunk_indices = np.arange(math.ceil((stop - start) / self._chunk_size))
        if self.shuffle:
            rng.shuffle(chunk_indices)
        n_chunks, pivot_index = len(chunk_indices), chunk_indices[-1]
        remainder = (stop - start) % self._chunk_size
        offsets = np.ones(n_chunks + 1, dtype=int) * self._chunk_size
        offsets[0] = start
        offsets[pivot_index + 1] = remainder if remainder else self._chunk_size
        offsets = np.cumsum(offsets)
        chunk_starts = offsets[:-1][chunk_indices]
        chunk_stops = offsets[1:][chunk_indices]
        return chunk_starts, chunk_stops

    def _resolve_start_stop(self, n_obs: int) -> tuple[int, int]:
        return self._mask.start or 0, self._mask.stop or n_obs


class ChunkSamplerWithReplacement(ChunkSampler):
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
        super().__init__(
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
            batch_size=batch_size,
            mask=mask,
            shuffle=True,
            drop_last=False,
            rng=rng,
        )
        self._n_iters = n_iters

    def validate(self, n_obs: int) -> None:
        super().validate(n_obs)
        start, stop = self._resolve_start_stop(n_obs)
        if stop - start < self._chunk_size:
            raise ValueError(
                f"Observation range ({stop - start}) is smaller than chunk_size ({self._chunk_size}). "
                "Reduce chunk_size or expand the mask range."
            )

    def _validate_worker_mode(self, worker_info: WorkerInfo | None) -> None:
        if worker_info is not None and worker_info.num_workers > 1:
            raise ValueError("Multiple workers are not supported with this sampler.")

    def n_iters(self, n_obs: int) -> int:
        return self._n_iters

    def _compute_chunks(self, start: int, stop: int, rng: np.random.Generator) -> _ChunkInfo:
        """Draw random contiguous chunks with replacement from the observation range."""
        # stop - start >= chunk_size is guaranteed by validate()
        chunk_starts = rng.integers(
            start, stop - self._chunk_size + 1, size=math.ceil((self._n_iters * self.batch_size) / self._chunk_size)
        )
        return chunk_starts, chunk_starts + self._chunk_size

    def _iter_from_chunks(
        self, chunk_info: _ChunkInfo, batch_rng: np.random.Generator, worker_info: WorkerInfo | None
    ) -> Iterator[LoadRequest]:
        load_requests = super()._iter_from_chunks(chunk_info, batch_rng, worker_info)
        batches_per_request = self._in_memory_size // self.batch_size
        n_full, tail = divmod(self._n_iters, batches_per_request)
        yield from itertools.islice(load_requests, n_full)
        if tail > 0:
            load_request = next(load_requests)
            yield {
                "starts": load_request["starts"],
                "stops": load_request["stops"],
                "splits": load_request["splits"][:tail],
            }

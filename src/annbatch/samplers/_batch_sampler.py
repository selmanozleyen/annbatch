"""Chunk batch sampler -- groups chunks into batched LoadRequests."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from annbatch.abc import Sampler
from annbatch.samplers._chunk_sampler import ChunkSampler
from annbatch.samplers._random import RandomChunkSampler
from annbatch.samplers._utils import get_torch_worker_info
from annbatch.utils import _spawn_worker_rng, check_lt_1, split_given_size

if TYPE_CHECKING:
    from collections.abc import Iterator

    from annbatch.samplers._utils import WorkerInfo
    from annbatch.types import LoadRequest


class ChunkBatchSampler(Sampler):
    """Batch sampler that wraps a :class:`ChunkSampler`.

    Analogous to :class:`torch.utils.data.BatchSampler`: it consumes chunk
    slices from the wrapped sampler, groups them into preload-sized I/O
    requests, optionally shuffles the in-memory indices, and splits them
    into batches.

    Parameters
    ----------
    sampler
        The chunk-level sampler that decides chunk ordering.
    batch_size
        Number of observations per batch.
    drop_last
        Whether to drop the last incomplete batch.
    shuffle
        Whether to shuffle observation indices within each loaded chunk
        group before splitting into batches.
    rng
        NumPy random generator for in-memory shuffling.
    """

    _sampler: ChunkSampler
    _batch_size: int
    _drop_last: bool
    _shuffle: bool
    _rng: np.random.Generator

    def __init__(
        self,
        sampler: ChunkSampler,
        batch_size: int,
        *,
        drop_last: bool = False,
        shuffle: bool = False,
        rng: np.random.Generator | None = None,
    ):
        check_lt_1([batch_size], ["batch_size"])
        preload_size = sampler.in_memory_size
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
        self._sampler = sampler
        self._batch_size = batch_size
        self._drop_last = drop_last
        self._shuffle = shuffle
        self._rng = rng or np.random.default_rng()

    @property
    def chunk_sampler(self) -> ChunkSampler:
        """The underlying chunk-level sampler."""
        return self._sampler

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def drop_last(self) -> bool:
        return self._drop_last

    @property
    def shuffle(self) -> bool:
        return self._shuffle

    def n_iters(self, n_obs: int) -> int:
        ns = self._effective_n_obs(n_obs)
        if self._drop_last:
            return ns // self._batch_size
        return math.ceil(ns / self._batch_size)

    def validate(self, n_obs: int) -> None:
        self._sampler.validate(n_obs)

    def _effective_n_obs(self, n_obs: int) -> int:
        """Number of observations that will actually be yielded."""
        if isinstance(self._sampler, RandomChunkSampler) and self._sampler._num_samples is not None:
            return self._sampler._num_samples
        start, stop = self._sampler._resolve_start_stop(n_obs)
        return stop - start

    def _sample(self, n_obs: int) -> Iterator[LoadRequest]:
        worker_info = get_torch_worker_info()
        self._validate_worker_mode(worker_info)

        chunks = self._sampler.chunks(n_obs)
        worker_rng = self._rng if worker_info is None else _spawn_worker_rng(self._rng, worker_info.id)

        if worker_info is not None:
            chunks = list(np.array_split(chunks, worker_info.num_workers)[worker_info.id])

        yield from self._iter_from_chunks(chunks, worker_rng)

    def _validate_worker_mode(self, worker_info: WorkerInfo | None) -> None:
        if worker_info is None or worker_info.num_workers <= 1:
            return
        if isinstance(self._sampler, RandomChunkSampler) and self._sampler.replacement:
            raise ValueError("Multiple workers are not supported with replacement sampling.")
        if not self._drop_last and self._batch_size != 1:
            raise ValueError("When using DataLoader with multiple workers drop_last=False is not supported.")

    def _iter_from_chunks(
        self,
        chunks: list[slice],
        batch_rng: np.random.Generator,
    ) -> Iterator[LoadRequest]:
        preload_nchunks = self._sampler.preload_nchunks
        in_memory_size = self._sampler.in_memory_size
        chunks_per_request = split_given_size(chunks, preload_nchunks)

        batch_indices = np.arange(in_memory_size)
        split_batch_indices = split_given_size(batch_indices, self._batch_size)

        for request_chunks in chunks_per_request[:-1]:
            if self._shuffle:
                batch_rng.shuffle(batch_indices)
                split_batch_indices = split_given_size(batch_indices, self._batch_size)
            yield {"chunks": request_chunks, "splits": split_batch_indices}

        final_chunks = chunks_per_request[-1]
        total_obs = int(sum(s.stop - s.start for s in final_chunks))
        if total_obs == 0:  # pragma: no cover
            raise RuntimeError("Last batch was found to have no observations. Please open an issue.")
        if self._drop_last:
            if total_obs < self._batch_size:
                return
            total_obs -= total_obs % self._batch_size
        indices = batch_rng.permutation(total_obs) if self._shuffle else np.arange(total_obs)
        yield {"chunks": final_chunks, "splits": split_given_size(indices, self._batch_size)}

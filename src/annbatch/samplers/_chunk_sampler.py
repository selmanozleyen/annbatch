"""Chunk-based sampler classes following a two-level design.

Level 1 -- chunk samplers (element-level):
    Decide *which* chunks to read and in *what order*.

Level 2 -- batch sampler:
    Groups chunks into preload-sized requests, handles in-memory
    shuffling, batch splitting, and ``drop_last``.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from annbatch.abc import Sampler
from annbatch.samplers._utils import get_torch_worker_info
from annbatch.utils import _spawn_worker_rng, check_lt_1, split_given_size

if TYPE_CHECKING:
    from collections.abc import Iterator

    from annbatch.samplers._utils import WorkerInfo
    from annbatch.types import LoadRequest


# ---------------------------------------------------------------------------
# Level 1: Chunk samplers -- decide chunk order
# ---------------------------------------------------------------------------


class ChunkSampler:
    """Base class for chunk-level samplers.

    A chunk sampler knows the chunk geometry (``chunk_size``,
    ``preload_nchunks``) and an observation mask, and produces an ordered
    list of chunk slices via :meth:`chunks`.  It does **not** know about
    batch size, ``drop_last``, or in-memory shuffling -- those belong to
    :class:`ChunkBatchSampler`.

    Parameters
    ----------
    chunk_size
        Size of each contiguous chunk read from disk.
    preload_nchunks
        Number of chunks loaded per I/O request.
    mask
        Observation range ``[start, stop)`` to sample from.
    rng
        NumPy random generator.
    """

    _chunk_size: int
    _preload_nchunks: int
    _mask: slice
    _rng: np.random.Generator

    def __init__(
        self,
        chunk_size: int,
        preload_nchunks: int,
        *,
        mask: slice | None = None,
        rng: np.random.Generator | None = None,
    ):
        check_lt_1([chunk_size, preload_nchunks], ["Chunk size", "Preloaded chunks"])

        if mask is None:
            mask = slice(0, None)
        if mask.step is not None and mask.step != 1:
            raise ValueError(f"mask.step must be 1, but got {mask.step}")
        start, stop = mask.start or 0, mask.stop
        if start < 0:
            raise ValueError("mask.start must be >= 0")
        if stop is not None and start >= stop:
            raise ValueError("mask.start must be < mask.stop when mask.stop is specified")

        self._chunk_size = chunk_size
        self._preload_nchunks = preload_nchunks
        self._mask = slice(start, stop)
        self._rng = rng or np.random.default_rng()

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def preload_nchunks(self) -> int:
        return self._preload_nchunks

    @property
    def mask(self) -> slice:
        return self._mask

    @mask.setter
    def mask(self, value: slice) -> None:
        self._mask = value

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    @rng.setter
    def rng(self, value: np.random.Generator) -> None:
        self._rng = value

    @property
    def in_memory_size(self) -> int:
        return self._chunk_size * self._preload_nchunks

    def n_chunks(self, n_obs: int) -> int:
        """Total number of chunks this sampler will produce."""
        start, stop = self._resolve_start_stop(n_obs)
        return self._n_chunks(start, stop)

    def chunks(self, n_obs: int) -> list[slice]:
        """Compute the ordered list of chunk slices for the given ``n_obs``."""
        start, stop = self._resolve_start_stop(n_obs)
        return self._compute_chunks(start, stop)

    def validate(self, n_obs: int) -> None:
        """Validate against the total number of observations."""
        start, stop = self._resolve_start_stop(n_obs)
        if stop > n_obs:
            raise ValueError(
                f"Sampler mask.stop ({stop}) exceeds loader n_obs ({n_obs}). "
                "The sampler range must be within the loader's observations."
            )
        if start >= stop:
            raise ValueError(f"Sampler mask.start ({start}) must be < mask.stop ({stop}).")

    @abstractmethod
    def _n_chunks(self, start: int, stop: int) -> int:
        """Return the number of chunks for the resolved range."""

    @abstractmethod
    def _compute_chunks(self, start: int, stop: int) -> list[slice]:
        """Return ordered chunk slices covering ``[start, stop)``."""

    def _resolve_start_stop(self, n_obs: int) -> tuple[int, int]:
        return self._mask.start or 0, self._mask.stop or n_obs


class SequentialChunkSampler(ChunkSampler):
    """Chunk sampler that yields chunks in sequential (deterministic) order.

    Analogous to :class:`torch.utils.data.SequentialSampler`.

    Parameters
    ----------
    chunk_size
        Size of each contiguous chunk.
    preload_nchunks
        Number of chunks per I/O request.
    mask
        Observation range ``[start, stop)`` to sample from.
    rng
        NumPy random generator (only used for distributed seeding).
    """

    def _n_chunks(self, start: int, stop: int) -> int:
        return math.ceil((stop - start) / self._chunk_size)

    def _compute_chunks(self, start: int, stop: int) -> list[slice]:
        n = self._n_chunks(start, stop)
        return [
            slice(
                start + i * self._chunk_size,
                min(start + (i + 1) * self._chunk_size, stop),
            )
            for i in range(n)
        ]


class RandomChunkSampler(ChunkSampler):
    """Chunk sampler that yields chunks in random order.

    Analogous to :class:`torch.utils.data.RandomSampler`.

    When ``replacement`` is *False* (default) the sampler produces a
    shuffled permutation of all chunks covering ``[start, stop)``.
    If ``num_samples`` exceeds the number of observations in one epoch,
    multiple full permutations are chained (same as PyTorch's
    ``RandomSampler``).

    When ``replacement`` is *True* the sampler draws random contiguous
    chunks independently.  ``num_samples`` is required in this mode.

    Parameters
    ----------
    chunk_size
        Size of each contiguous chunk.
    preload_nchunks
        Number of chunks per I/O request.
    replacement
        Whether to sample chunks with replacement.
    num_samples
        Total number of *observations* to draw. Defaults to one full epoch
        (all observations in the mask range) when ``replacement=False``.
        Required when ``replacement=True``.
    mask
        Observation range ``[start, stop)`` to sample from.
    rng
        NumPy random generator for reproducibility.
    """

    _replacement: bool
    _num_samples: int | None

    def __init__(
        self,
        chunk_size: int,
        preload_nchunks: int,
        *,
        replacement: bool = False,
        num_samples: int | None = None,
        mask: slice | None = None,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(chunk_size=chunk_size, preload_nchunks=preload_nchunks, mask=mask, rng=rng)
        if replacement and num_samples is None:
            raise ValueError("num_samples is required when replacement=True")
        if num_samples is not None:
            check_lt_1([num_samples], ["num_samples"])
        self._replacement = replacement
        self._num_samples = num_samples

    @property
    def replacement(self) -> bool:
        return self._replacement

    def num_samples(self, n_obs: int) -> int:
        """Effective number of observations to draw."""
        if self._num_samples is not None:
            return self._num_samples
        start, stop = self._resolve_start_stop(n_obs)
        return stop - start

    def validate(self, n_obs: int) -> None:
        super().validate(n_obs)
        if self._replacement:
            start, stop = self._resolve_start_stop(n_obs)
            if stop - start < self._chunk_size:
                raise ValueError(
                    f"Observation range ({stop - start}) is smaller than chunk_size ({self._chunk_size}). "
                    "Reduce chunk_size or expand the mask range."
                )

    def _n_chunks(self, start: int, stop: int) -> int:
        if self._replacement:
            ns = self._num_samples  # guaranteed non-None by __init__
            return math.ceil(ns / self._chunk_size)

        epoch_chunks = math.ceil((stop - start) / self._chunk_size)
        ns = self._num_samples if self._num_samples is not None else stop - start
        return math.ceil(ns / self._chunk_size) if ns != stop - start else epoch_chunks

    def _compute_chunks(self, start: int, stop: int) -> list[slice]:
        if self._replacement:
            return self._compute_chunks_with_replacement(start, stop)
        return self._compute_chunks_without_replacement(start, stop)

    def _compute_chunks_with_replacement(self, start: int, stop: int) -> list[slice]:
        n = self._n_chunks(start, stop)
        starts = self._rng.integers(start, stop - self._chunk_size + 1, size=n)
        return [slice(int(s), int(s + self._chunk_size)) for s in starts]

    def _compute_chunks_without_replacement(self, start: int, stop: int) -> list[slice]:
        """Shuffled epoch(s), chaining multiple permutations like torch RandomSampler."""
        epoch_n_chunks = math.ceil((stop - start) / self._chunk_size)
        total_chunks_needed = self._n_chunks(start, stop)

        full_epochs, tail = divmod(total_chunks_needed, epoch_n_chunks)

        chunks: list[slice] = []
        for _ in range(full_epochs):
            chunks.extend(self._one_epoch_permuted(start, stop, epoch_n_chunks))
        if tail > 0:
            chunks.extend(self._one_epoch_permuted(start, stop, epoch_n_chunks)[:tail])
        return chunks

    def _one_epoch_permuted(self, start: int, stop: int, n_chunks: int) -> list[slice]:
        """One shuffled pass over all chunks, incomplete chunk placed randomly."""
        chunk_indices = self._rng.permutation(n_chunks)
        pivot_index = chunk_indices[-1]
        offsets = np.ones(n_chunks + 1, dtype=int) * self._chunk_size
        offsets[0] = start
        incomplete = (stop - start) % self._chunk_size
        offsets[pivot_index + 1] = incomplete if incomplete else self._chunk_size
        offsets = np.cumsum(offsets)
        starts_arr = offsets[:-1][chunk_indices]
        stops_arr = offsets[1:][chunk_indices]
        return [slice(int(s), int(e)) for s, e in zip(starts_arr, stops_arr, strict=True)]


# ---------------------------------------------------------------------------
# Level 2: Batch sampler -- groups chunks, handles batching & shuffle
# ---------------------------------------------------------------------------


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

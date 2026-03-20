"""Random chunk sampler."""

from __future__ import annotations

import math

import numpy as np

from annbatch.samplers._chunk_sampler import ChunkSampler
from annbatch.utils import check_lt_1


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

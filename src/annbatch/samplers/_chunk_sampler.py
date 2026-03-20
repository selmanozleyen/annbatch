"""Base class for chunk-level samplers."""

from __future__ import annotations

from abc import abstractmethod

import numpy as np

from annbatch.utils import check_lt_1


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

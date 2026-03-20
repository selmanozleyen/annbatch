"""Sequential chunk sampler."""

from __future__ import annotations

import math

from annbatch.samplers._chunk_sampler import ChunkSampler


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

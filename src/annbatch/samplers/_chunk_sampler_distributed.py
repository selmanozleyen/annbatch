"""Distributed chunk-based sampler."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from annbatch.abc import Sampler
from annbatch.utils import _spawn_worker_rng

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from annbatch.types import LoadRequest

from annbatch.samplers._chunk_sampler import ChunkBatchSampler


def _get_dist_info_torch() -> tuple[int, int]:
    """Get rank and world_size from ``torch.distributed``."""
    import torch.distributed as dist

    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed is not initialized. "
            "Initialize it before creating a ChunkSamplerDistributed with dist_info='torch'."
        )
    return dist.get_rank(), dist.get_world_size()


def _get_dist_info_jax() -> tuple[int, int]:
    """Get rank and world_size from JAX multi-process API."""
    import jax

    if not jax.distributed.is_initialized():
        raise RuntimeError(
            "JAX distributed is not initialized. "
            "Call jax.distributed.initialize() before creating a ChunkSamplerDistributed with dist_info='jax'."
        )
    return jax.process_index(), jax.process_count()


DISTRIBUTED_BACKENDS: dict[str, Callable[[], tuple[int, int]]] = {
    "torch": _get_dist_info_torch,
    "jax": _get_dist_info_jax,
}


class ChunkSamplerDistributed(Sampler):
    """Distributed chunk-based sampler that shards data across processes.

    Wraps a :class:`ChunkBatchSampler` and partitions the observation
    range across ``world_size`` processes.  Each rank receives a
    non-overlapping slice of the data.

    When ``enforce_equal_batches`` is *True* (the default), the per-rank
    observation count is rounded down to the nearest multiple of
    ``batch_size``, guaranteeing that every rank yields exactly the same
    number of complete batches.

    Parameters
    ----------
    sampler
        The batch sampler to distribute.
    dist_info
        How to obtain rank and world size.
        Either ``"torch"`` / ``"jax"``, or a callable returning
        ``(rank, world_size)``.
    enforce_equal_batches
        If *True*, round each rank's observation count down to a
        multiple of ``batch_size`` so all ranks yield the same
        number of batches.
    """

    _rank: int
    _world_size: int
    _enforce_equal_batches: bool
    _sampler: ChunkBatchSampler

    def __init__(
        self,
        sampler: ChunkBatchSampler,
        *,
        dist_info: Literal["torch", "jax"] | Callable[[], tuple[int, int]],
        enforce_equal_batches: bool = True,
    ):
        if callable(dist_info):
            self._rank, self._world_size = dist_info()
        elif dist_info in DISTRIBUTED_BACKENDS:
            self._rank, self._world_size = DISTRIBUTED_BACKENDS[dist_info]()
        else:
            raise ValueError(f"Unknown dist_info {dist_info!r}. Supported backends: {sorted(DISTRIBUTED_BACKENDS)}")
        self._enforce_equal_batches = enforce_equal_batches
        self._sampler = sampler
        sampler.chunk_sampler.rng = _spawn_worker_rng(sampler.chunk_sampler.rng, self._rank)

    @property
    def batch_size(self) -> int:
        return self._sampler.batch_size

    @property
    def shuffle(self) -> bool:
        return self._sampler.shuffle

    def _shard_mask(self, n_obs: int) -> slice:
        """Return the contiguous observation slice for this rank."""
        per_rank = n_obs // self._world_size
        if self._enforce_equal_batches:
            per_rank = per_rank // self._sampler.batch_size * self._sampler.batch_size
        rank_start = self._rank * per_rank
        rank_stop = rank_start + per_rank
        return slice(rank_start, rank_stop)

    def n_iters(self, n_obs: int) -> int:
        self._sampler.chunk_sampler.mask = self._shard_mask(n_obs)
        return self._sampler.n_iters(n_obs)

    def validate(self, n_obs: int) -> None:
        self._sampler.chunk_sampler.mask = self._shard_mask(n_obs)
        self._sampler.validate(n_obs)

    def _sample(self, n_obs: int) -> Iterator[LoadRequest]:
        self._sampler.chunk_sampler.mask = self._shard_mask(n_obs)
        yield from self._sampler._sample(n_obs)

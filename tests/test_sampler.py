"""Tests for SequentialChunkSampler, RandomChunkSampler, ChunkBatchSampler, and ChunkSamplerDistributed."""

from __future__ import annotations

import math
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from annbatch import (
    ChunkBatchSampler,
    ChunkSamplerDistributed,
    RandomChunkSampler,
    SequentialChunkSampler,
)
from annbatch.abc import Sampler
from annbatch.samplers._utils import WorkerInfo


def collect_indices(sampler: Sampler, n_obs: int) -> list[int]:
    """Helper to collect all indices from sampler."""
    indices: list[int] = []
    for load_request in sampler.sample(n_obs):
        assert len(load_request["splits"]) > 0, "splits must be non-empty"
        assert all(len(s) > 0 for s in load_request["splits"]), "splits must be non-empty"
        for s in load_request["chunks"]:
            indices.extend(range(s.start, s.stop))
    return indices


# =============================================================================
# Helpers to build samplers compactly
# =============================================================================


def _seq_batch_sampler(
    chunk_size: int,
    preload_nchunks: int,
    batch_size: int,
    *,
    mask: slice | None = None,
    drop_last: bool = False,
    rng: np.random.Generator | None = None,
) -> ChunkBatchSampler:
    cs = SequentialChunkSampler(chunk_size, preload_nchunks, mask=mask, rng=rng)
    return ChunkBatchSampler(cs, batch_size, drop_last=drop_last, shuffle=False, rng=rng)


def _rand_batch_sampler(
    chunk_size: int,
    preload_nchunks: int,
    batch_size: int,
    *,
    replacement: bool = False,
    num_samples: int | None = None,
    mask: slice | None = None,
    drop_last: bool = False,
    rng: np.random.Generator | None = None,
) -> ChunkBatchSampler:
    cs = RandomChunkSampler(
        chunk_size, preload_nchunks, replacement=replacement, num_samples=num_samples, mask=mask, rng=rng,
    )
    return ChunkBatchSampler(cs, batch_size, drop_last=drop_last, shuffle=True, rng=rng)


# =============================================================================
# Mask coverage tests
# =============================================================================


@pytest.mark.parametrize(
    ("n_obs", "chunk_size", "start", "stop", "batch_size", "preload_nchunks", "shuffle", "drop_last"),
    [
        # Basic full dataset
        pytest.param(100, 10, None, None, 5, 2, False, False, id="full_dataset"),
        # mask.start only
        pytest.param(100, 10, 30, None, 5, 2, False, False, id="start_at_chunk_boundary"),
        pytest.param(100, 10, 35, None, 5, 2, False, False, id="start_not_at_chunk_boundary"),
        pytest.param(120, 12, 90, None, 3, 1, False, False, id="start_near_end"),
        pytest.param(100, 10, 20, None, 5, 2, False, False, id="start_mask_stop_none"),
        # mask.stop only
        pytest.param(50, 10, None, 50, 5, 2, False, False, id="stop_at_chunk_boundary"),
        pytest.param(47, 10, None, 47, 5, 2, False, False, id="stop_not_at_chunk_boundary"),
        # Both bounds
        pytest.param(60, 10, 20, 60, 5, 2, False, False, id="both_at_chunk_boundaries"),
        pytest.param(67, 10, 23, 67, 5, 2, False, False, id="both_not_at_chunk_boundaries"),
        pytest.param(28, 10, 22, 28, 2, 1, False, False, id="single_chunk_span"),
        pytest.param(100, 10, 15, 85, 5, 2, False, False, id="both_non_aligned"),
        pytest.param(100, 10, 20, 80, 5, 2, False, False, id="both_aligned"),
        # Edge cases
        pytest.param(100, 10, 95, 100, 10, 1, False, False, id="very_small_mask"),
        # With shuffle
        pytest.param(100, 10, 30, None, 5, 2, True, False, id="shuffle_with_start"),
        pytest.param(75, 10, 25, 75, 5, 2, True, False, id="shuffle_with_both_bounds"),
        # drop_last edge cases: remainder less than batch_size
        pytest.param(45, 20, None, None, 10, 2, False, True, id="drop_last_remainder_less_than_batch"),
        pytest.param(5, 20, None, None, 10, 2, False, True, id="drop_last_total_less_than_batch"),
    ],
)
def test_mask_coverage(
    n_obs: int,
    chunk_size: int,
    start: int | None,
    stop: int | None,
    batch_size: int,
    preload_nchunks: int,
    shuffle: bool,
    drop_last: bool,
):
    """Test sampler covers exactly the expected range, and ordering is correct when not shuffled."""
    rng = np.random.default_rng(42) if shuffle else None
    if shuffle:
        sampler = _rand_batch_sampler(
            chunk_size, preload_nchunks, batch_size,
            mask=slice(start, stop), drop_last=drop_last, rng=rng,
        )
    else:
        sampler = _seq_batch_sampler(
            chunk_size, preload_nchunks, batch_size,
            mask=slice(start, stop), drop_last=drop_last, rng=rng,
        )

    expected_start = start if start is not None else 0
    expected_stop = stop if stop is not None else n_obs
    if drop_last:
        total_obs = expected_stop - expected_start
        expected_stop = expected_start + (total_obs // batch_size) * batch_size
    expected_indices = list(range(expected_start, expected_stop))

    all_indices = collect_indices(sampler, n_obs)

    if shuffle:
        assert set(all_indices) == set(expected_indices), "Sampler should cover all expected indices"
    else:
        assert all_indices == expected_indices, f"all_indices: {all_indices} != expected_indices: {expected_indices}"

    sampler.validate(n_obs)


def test_batch_sizes_match_expected_pattern():
    """Test that batch sizes match expected pattern."""
    n_obs, chunk_size, preload_nchunks, batch_size = 103, 10, 2, 5
    expected_last_chunk_size = 3
    expected_last_batch_size = 3
    expected_last_num_splits = 1
    expected_num_load_requests = 6
    sampler = _seq_batch_sampler(chunk_size, preload_nchunks, batch_size)

    all_requests = list(sampler.sample(n_obs))
    assert len(all_requests) == expected_num_load_requests
    for req_idx, load_request in enumerate(all_requests[:-1]):
        assert all(chunk.stop - chunk.start == chunk_size for chunk in load_request["chunks"]), (
            f"chunk size mismatch at request {req_idx}:",
            f"chunks: {load_request['chunks']}",
        )
        assert all(len(split) == batch_size for split in load_request["splits"]), (
            f"batch size mismatch at request {req_idx}:splits: {load_request['splits']}"
        )
    last_request = all_requests[-1]
    assert len(last_request["splits"]) == expected_last_num_splits, "last request num splits mismatch"
    assert all(chunk.stop - chunk.start == expected_last_chunk_size for chunk in last_request["chunks"]), (
        "last request chunk size mismatch",
        f"chunks: {last_request['chunks']}",
    )
    assert all(len(split) == expected_last_batch_size for split in last_request["splits"]), (
        "last request batch size mismatch",
        f"splits: {last_request['splits']}",
    )


# =============================================================================
# Worker tests
# =============================================================================


@pytest.mark.parametrize(
    ("n_obs", "chunk_size", "preload_nchunks", "batch_size", "num_workers", "drop_last"),
    [
        pytest.param(200, 10, 2, 10, 2, True, id="two_workers"),
        pytest.param(300, 10, 3, 10, 3, True, id="three_workers"),
        pytest.param(600, 10, 4, 1, 4, False, id="batch_size_one_torch_dataloader_case"),
        pytest.param(100, 10, 4, 1, 1, False, id="batch_size_one_single_worker_case"),
        pytest.param(95, 10, 4, 1, 1, False, id="batch_size_one_non_divisible_obs_case"),
        pytest.param(100, 10, 4, 1, 3, False, id="batch_size_one_three_workers_uneven_case"),
    ],
)
def test_workers_cover_full_dataset_without_overlap(
    n_obs: int, chunk_size: int, preload_nchunks: int, batch_size: int, num_workers: int, drop_last: bool
):
    """Test workers cover full dataset without overlap."""
    all_worker_indices = []
    for worker_id in range(num_workers):
        sampler = _seq_batch_sampler(
            chunk_size, preload_nchunks, batch_size, drop_last=drop_last,
        )
        with patch(
            "annbatch.samplers._batch_sampler.get_torch_worker_info",
            return_value=WorkerInfo(id=worker_id, num_workers=num_workers),
        ):
            all_worker_indices.append(collect_indices(sampler, n_obs))

    for i in range(num_workers):
        for j in range(i + 1, num_workers):
            assert set(all_worker_indices[i]).isdisjoint(all_worker_indices[j])

    assert set().union(*all_worker_indices) == set(range(n_obs))


@pytest.mark.parametrize(
    "make_sampler",
    [
        pytest.param(
            lambda seed: _rand_batch_sampler(10, 2, 5, rng=np.random.default_rng(seed)),
            id="without_replacement",
        ),
        pytest.param(
            lambda seed: _rand_batch_sampler(10, 2, 5, replacement=True, num_samples=100, rng=np.random.default_rng(seed)),
            id="with_replacement",
        ),
    ],
)
def test_batch_shuffle_is_reproducible_with_same_seed_rng(make_sampler):
    """Test that sampling is fully reproducible with the same seed and differs with another."""
    n_obs = 100

    indices1 = collect_indices(make_sampler(42), n_obs)
    indices2 = collect_indices(make_sampler(42), n_obs)
    indices3 = collect_indices(make_sampler(99), n_obs)

    assert indices1 == indices2, "Sampling should be reproducible with same seed"
    assert indices1 != indices3, "Different seeds should produce different results"


# =============================================================================
# Validation tests
# =============================================================================


@pytest.mark.parametrize(
    ("mask", "n_obs", "error_match"),
    [
        pytest.param(slice(0, 100), 100, None, id="valid_config"),
        pytest.param(slice(0, 200), 100, "mask.stop.*exceeds loader n_obs", id="stop_exceeds_n_obs"),
    ],
)
def test_validate(mask: slice, n_obs: int, error_match: str | None):
    """Test validate behavior for various configurations."""
    sampler = _seq_batch_sampler(10, 2, 5, mask=mask)
    if error_match:
        with pytest.raises(ValueError, match=error_match):
            sampler.validate(n_obs=n_obs)
    else:
        sampler.validate(n_obs=n_obs)


@pytest.mark.parametrize(
    "make_chunk_sampler",
    [
        pytest.param(lambda **kw: SequentialChunkSampler(**kw), id="sequential"),
        pytest.param(lambda **kw: RandomChunkSampler(**kw), id="random"),
    ],
)
@pytest.mark.parametrize(
    ("mask", "error_match"),
    [
        pytest.param(slice(-1, 100), "mask.start must be >= 0", id="negative_start"),
        pytest.param(slice(50, 50), "mask.start must be < mask.stop", id="start_equals_stop"),
        pytest.param(slice(100, 50), "mask.start must be < mask.stop", id="start_greater_than_stop"),
        pytest.param(slice(0, 100, 2), "mask.step must be 1, but got 2", id="step_not_one"),
    ],
)
def test_invalid_mask_raises(make_chunk_sampler, mask: slice, error_match: str):
    """Test that invalid mask configurations raise ValueError at construction."""
    with pytest.raises(ValueError, match=error_match):
        make_chunk_sampler(chunk_size=10, preload_nchunks=2, mask=mask)


@pytest.mark.parametrize(
    ("kwargs", "n_obs", "error_match"),
    [
        pytest.param({"num_samples": 0}, None, "num_samples", id="num_samples_zero"),
        pytest.param({"num_samples": -1}, None, "num_samples", id="num_samples_negative"),
        pytest.param(
            {"replacement": True, "num_samples": 30}, 5, "smaller than chunk_size",
            id="n_obs_smaller_than_chunk",
        ),
        pytest.param(
            {"replacement": True, "num_samples": 30, "mask": slice(50, 55)},
            100,
            "smaller than chunk_size",
            id="mask_range_smaller_than_chunk",
        ),
    ],
)
def test_invalid_random_sampler(kwargs: dict, n_obs: int | None, error_match: str):
    """Test that invalid configurations raise ValueError for RandomChunkSampler."""
    defaults = {"chunk_size": 10, "preload_nchunks": 2}
    with pytest.raises(ValueError, match=error_match):
        cs = RandomChunkSampler(**(defaults | kwargs))
        if n_obs is not None:
            bs = ChunkBatchSampler(cs, 5, shuffle=True)
            list(bs.sample(n_obs))


def test_replacement_requires_num_samples():
    """Test that replacement=True without num_samples raises."""
    with pytest.raises(ValueError, match="num_samples is required"):
        RandomChunkSampler(chunk_size=10, preload_nchunks=2, replacement=True)


@pytest.mark.parametrize(
    ("n_obs", "chunk_size", "preload_nchunks", "batch_size", "num_samples", "mask"),
    [
        pytest.param(100, 10, 2, 5, 100, slice(0, None), id="basic"),
        pytest.param(100, 10, 2, 5, 500, slice(0, None), id="more_samples_than_obs"),
        pytest.param(100, 10, 2, 5, 10, slice(0, None), id="single_chunk_worth"),
        pytest.param(100, 10, 2, 5, 100, slice(20, 80), id="with_mask"),
        pytest.param(103, 10, 2, 5, 200, slice(0, None), id="non_divisible_obs"),
        pytest.param(100, 10, 2, 5, 70, slice(0, None), id="partial_epoch"),
    ],
)
def test_replacement_invariants(
    n_obs: int, chunk_size: int, preload_nchunks: int, batch_size: int, num_samples: int, mask: slice
):
    """Test RandomChunkSampler with replacement yields correct chunk count, chunks within bounds, uniform chunk sizes."""
    start = mask.start or 0
    stop = mask.stop or n_obs
    sampler = _rand_batch_sampler(
        chunk_size, preload_nchunks, batch_size,
        replacement=True, num_samples=num_samples, mask=mask, rng=np.random.default_rng(42),
    )
    count = 0
    for load_request in sampler.sample(n_obs):
        assert len(load_request["chunks"]) > 0, "Load request must have at least one chunk"
        for chunk in load_request["chunks"]:
            assert chunk.stop - chunk.start == chunk_size, f"Non-uniform chunk: {chunk}"
            assert chunk.start >= start, f"Chunk start {chunk.start} < mask start {start}"
            assert chunk.stop <= stop, f"Chunk stop {chunk.stop} > mask stop {stop}"
        count += len(load_request["splits"])
    expected_batches = math.ceil(num_samples / batch_size)
    assert count == expected_batches, f"Expected {expected_batches} batches, got {count}"


def test_replacement_with_multiple_workers_raises():
    """Test that replacement sampling raises when used with multiple workers."""
    sampler = _rand_batch_sampler(
        10, 2, 5, replacement=True, num_samples=100, rng=np.random.default_rng(42),
    )
    with (
        patch(
            "annbatch.samplers._batch_sampler.get_torch_worker_info",
            return_value=WorkerInfo(id=0, num_workers=2),
        ),
        pytest.raises(ValueError, match="Multiple workers are not supported with replacement"),
    ):
        list(sampler.sample(100))


@pytest.mark.parametrize(
    ("sampler", "n_obs", "expected"),
    [
        pytest.param(
            _rand_batch_sampler(10, 2, 5, replacement=True, num_samples=250, rng=np.random.default_rng(42)),
            100,
            50,
            id="replacement_returns_based_on_num_samples",
        ),
        pytest.param(
            _seq_batch_sampler(10, 2, 5),
            100,
            20,
            id="sequential_full_epoch",
        ),
        pytest.param(
            _seq_batch_sampler(10, 2, 5, drop_last=True),
            100,
            20,
            id="sequential_drop_last_exact",
        ),
        pytest.param(
            _seq_batch_sampler(10, 2, 5),
            103,
            21,
            id="sequential_ceil",
        ),
        pytest.param(
            _seq_batch_sampler(10, 2, 5, drop_last=True),
            103,
            20,
            id="sequential_drop_last_floor",
        ),
    ],
)
def test_n_iters_property(sampler: Sampler, n_obs: int, expected: int):
    """Test that n_iters() returns the correct value for different configurations."""
    assert sampler.n_iters(n_obs) == expected


# =============================================================================
# n_obs change tests (To verify nothing is cached between calls.)
# =============================================================================


@pytest.mark.parametrize(
    ("n_obs_values", "expected_ranges"),
    [
        pytest.param([50, 100], [range(50), range(100)], id="increase_changes_result"),
        pytest.param([100, 100], [range(100), range(100)], id="same_gives_same_coverage"),
    ],
)
def test_n_obs_coverage(n_obs_values: list[int], expected_ranges: list[range]):
    """Test that n_obs changes affect sampling results appropriately."""
    sampler = _seq_batch_sampler(10, 2, 5)

    results = [collect_indices(sampler, n) for n in n_obs_values]

    for result, expected in zip(results, expected_ranges, strict=True):
        assert result == list(expected), f"result: {result} != expected: {expected}"


# =============================================================================
# Multi-epoch tests (num_samples > n_obs, replacement=False)
# =============================================================================


def test_multi_epoch_without_replacement():
    """Test that num_samples > n_obs chains multiple shuffled epochs."""
    n_obs, chunk_size, preload_nchunks, batch_size = 100, 10, 2, 5
    num_samples = 250

    cs = RandomChunkSampler(chunk_size, preload_nchunks, num_samples=num_samples, rng=np.random.default_rng(42))
    sampler = ChunkBatchSampler(cs, batch_size, shuffle=True, rng=np.random.default_rng(42))

    all_indices = collect_indices(sampler, n_obs)
    assert len(all_indices) == num_samples, f"Expected {num_samples} indices, got {len(all_indices)}"
    expected_batches = math.ceil(num_samples / batch_size)
    assert sampler.n_iters(n_obs) == expected_batches


# =============================================================================
# ChunkSamplerDistributed tests
# =============================================================================


def _make_distributed_sampler_torch(
    rank: int, world_size: int, *, enforce_equal_batches: bool = True, **sampler_kwargs
) -> ChunkSamplerDistributed:
    """Create a ChunkSamplerDistributed with mocked torch.distributed backend."""
    mock_dist = MagicMock()
    mock_dist.is_initialized.return_value = True
    mock_dist.get_rank.return_value = rank
    mock_dist.get_world_size.return_value = world_size
    mock_torch = MagicMock()
    mock_torch.distributed = mock_dist
    batch_sampler = _seq_batch_sampler(**sampler_kwargs)
    with patch.dict(sys.modules, {"torch": mock_torch, "torch.distributed": mock_dist}):
        return ChunkSamplerDistributed(batch_sampler, dist_info="torch", enforce_equal_batches=enforce_equal_batches)


def _make_distributed_sampler_jax(
    rank: int, world_size: int, *, enforce_equal_batches: bool = True, **sampler_kwargs
) -> ChunkSamplerDistributed:
    """Create a ChunkSamplerDistributed with mocked jax backend."""
    mock_jax = MagicMock()
    mock_jax.process_index.return_value = rank
    mock_jax.process_count.return_value = world_size
    mock_jax.distributed.is_initialized.return_value = True
    batch_sampler = _seq_batch_sampler(**sampler_kwargs)
    with patch.dict(sys.modules, {"jax": mock_jax}):
        return ChunkSamplerDistributed(batch_sampler, dist_info="jax", enforce_equal_batches=enforce_equal_batches)


_SAMPLER_FACTORIES = {
    "torch": _make_distributed_sampler_torch,
    "jax": _make_distributed_sampler_jax,
}


@pytest.fixture(params=["torch", "jax"])
def make_distributed_sampler(request):
    """Fixture that yields a sampler factory for each backend."""
    return _SAMPLER_FACTORIES[request.param]


class TestChunkSamplerDistributed:
    """Tests for ChunkSamplerDistributed, parameterized over all backends."""

    def test_not_initialized_raises_torch(self):
        """RuntimeError when torch.distributed is not initialized."""
        mock_dist = MagicMock()
        mock_dist.is_initialized.return_value = False
        mock_torch = MagicMock()
        mock_torch.distributed = mock_dist
        sampler = _seq_batch_sampler(10, 2, 10)
        with patch.dict(sys.modules, {"torch": mock_torch, "torch.distributed": mock_dist}):
            with pytest.raises(RuntimeError, match="torch.distributed is not initialized"):
                ChunkSamplerDistributed(sampler, dist_info="torch")

    def test_not_initialized_raises_jax(self):
        """RuntimeError when jax.distributed is not initialized."""
        mock_jax = MagicMock()
        mock_jax.distributed.is_initialized.return_value = False
        sampler = _seq_batch_sampler(10, 2, 10)
        with patch.dict(sys.modules, {"jax": mock_jax}):
            with pytest.raises(RuntimeError, match="JAX distributed is not initialized"):
                ChunkSamplerDistributed(sampler, dist_info="jax")

    def test_unknown_dist_info_raises(self):
        """ValueError for an unsupported dist_info string."""
        sampler = _seq_batch_sampler(10, 2, 10)
        with pytest.raises(ValueError, match="Unknown dist_info"):
            ChunkSamplerDistributed(sampler, dist_info="mpi")

    def test_shards_are_disjoint_and_cover_full_dataset(self, make_distributed_sampler):
        """All ranks receive non-overlapping shards that together cover the full dataset."""
        n_obs, world_size = 200, 4
        chunk_size, preload_nchunks, batch_size = 10, 2, 10

        all_indices = []
        for rank in range(world_size):
            sampler = make_distributed_sampler(
                rank=rank,
                world_size=world_size,
                chunk_size=chunk_size,
                preload_nchunks=preload_nchunks,
                batch_size=batch_size,
            )
            all_indices.append(collect_indices(sampler, n_obs))

        for i in range(world_size):
            for j in range(i + 1, world_size):
                assert set(all_indices[i]).isdisjoint(set(all_indices[j]))

        assert set().union(*all_indices) == set(range(n_obs))

    @pytest.mark.parametrize(
        "n_obs,world_size,batch_size,chunk_size,preload_nchunks",
        [
            pytest.param(200, 4, 10, 10, 2, id="evenly_divisible"),
            pytest.param(205, 3, 10, 10, 2, id="remainder_obs"),
            pytest.param(1000, 7, 5, 10, 2, id="prime_world_size"),
            pytest.param(100, 3, 5, 10, 2, id="small_dataset"),
        ],
    )
    def test_enforce_equal_batches_all_ranks_same_count(
        self, make_distributed_sampler, n_obs, world_size, batch_size, chunk_size, preload_nchunks
    ):
        """enforce_equal_batches=True guarantees identical batch counts across ranks."""
        batch_counts = []
        for rank in range(world_size):
            sampler = make_distributed_sampler(
                rank=rank,
                world_size=world_size,
                chunk_size=chunk_size,
                preload_nchunks=preload_nchunks,
                batch_size=batch_size,
                enforce_equal_batches=True,
            )
            n_batches = sum(len(lr["splits"]) for lr in sampler.sample(n_obs))
            batch_counts.append(n_batches)

        assert len(set(batch_counts)) == 1, f"Batch counts differ across ranks: {batch_counts}"

    @pytest.mark.parametrize(
        ("enforce_equal_batches", "expected"),
        [(True, 30), (False, 35)],
        ids=["rounded", "raw"],
    )
    def test_enforce_equal_batches_per_rank_count(self, make_distributed_sampler, enforce_equal_batches, expected):
        """enforce_equal_batches controls whether per_rank is rounded down to a multiple of batch_size."""
        n_obs, world_size = 107, 3
        chunk_size, preload_nchunks, batch_size = 10, 1, 10
        sampler = make_distributed_sampler(
            rank=0,
            world_size=world_size,
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
            batch_size=batch_size,
            enforce_equal_batches=enforce_equal_batches,
        )
        indices = collect_indices(sampler, n_obs)
        assert len(set(indices)) == expected

    def test_batch_shuffle_is_reproducible_with_same_seed_rng(self, make_distributed_sampler):
        """Test that batch shuffling is reproducible when passing in rngs with identical seeds."""
        n_obs, chunk_size, preload_nchunks, batch_size = 200, 10, 2, 5
        world_size = 4
        seed = 42

        def collect_splits(sampler: ChunkSamplerDistributed) -> list[list[int]]:
            all_splits: list[list[int]] = []
            for load_request in sampler.sample(n_obs):
                for split in load_request["splits"]:
                    all_splits.append(split.tolist())
            return all_splits

        def _make_rand_dist_sampler(rank, ws, *, rng, **kw):
            cs = RandomChunkSampler(chunk_size, preload_nchunks, rng=rng)
            bs = ChunkBatchSampler(cs, batch_size, shuffle=True, rng=rng)
            mock_dist = MagicMock()
            mock_dist.is_initialized.return_value = True
            mock_dist.get_rank.return_value = rank
            mock_dist.get_world_size.return_value = ws
            mock_torch = MagicMock()
            mock_torch.distributed = mock_dist
            with patch.dict(sys.modules, {"torch": mock_torch, "torch.distributed": mock_dist}):
                return ChunkSamplerDistributed(bs, dist_info="torch")

        splits_per_run: list[dict[int, list[list[int]]]] = []
        for _ in range(3):
            splits_by_rank: dict[int, list[list[int]]] = {}
            for rank in range(world_size):
                sampler = _make_rand_dist_sampler(
                    rank, world_size, rng=np.random.default_rng(seed),
                )
                splits_by_rank[rank] = collect_splits(sampler)
            splits_per_run.append(splits_by_rank)

        for rank in range(world_size):
            assert splits_per_run[0][rank] == splits_per_run[1][rank], (
                f"Rank {rank}: batch shuffling should be reproducible with same seed"
            )

    def test_n_iters_matches_actual_batch_count(self, make_distributed_sampler):
        """n_iters should match the actual number of yielded batches."""
        n_obs, world_size = 205, 3
        chunk_size, preload_nchunks, batch_size = 10, 2, 10

        for rank in range(world_size):
            sampler = make_distributed_sampler(
                rank=rank,
                world_size=world_size,
                chunk_size=chunk_size,
                preload_nchunks=preload_nchunks,
                batch_size=batch_size,
                enforce_equal_batches=True,
                drop_last=True,
            )
            expected = sampler.n_iters(n_obs)
            actual = sum(len(lr["splits"]) for lr in sampler.sample(n_obs))
            assert actual == expected, f"rank {rank}: n_iters={expected}, actual={actual}"

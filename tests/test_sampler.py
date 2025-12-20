"""Tests for SliceSampler."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from annbatch.sampler import SliceSampler


class TestSliceSamplerBasic:
    """Tests for basic SliceSampler functionality."""

    def test_full_dataset(self):
        """Test sampler covers full dataset when no start/end specified."""
        n_obs = 100
        chunk_size = 10
        preload_nchunks = 2
        batch_size = 5

        sampler = SliceSampler(
            n_obs=n_obs,
            batch_size=batch_size,
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
        )

        all_indices = set()
        for slices, _splits, _ in sampler:
            for s in slices:
                all_indices.update(range(s.start, s.stop))

        assert all_indices == set(range(n_obs))

    def test_batch_sizes(self):
        """Test that batches have correct size (except possibly last)."""
        n_obs = 100
        chunk_size = 10
        preload_nchunks = 2
        batch_size = 7

        sampler = SliceSampler(
            n_obs=n_obs,
            batch_size=batch_size,
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
        )

        batch_sizes = []
        for _, splits, _ in sampler:
            for split in splits:
                batch_sizes.append(len(split))

        # All but last should be batch_size
        assert all(bs == batch_size for bs in batch_sizes[:-1])
        # Last batch should be <= batch_size
        assert batch_sizes[-1] <= batch_size


class TestSliceSamplerStartIndex:
    """Tests for SliceSampler with non-zero start_index."""

    def test_start_index_at_chunk_boundary(self):
        """Test start_index aligned with chunk boundary."""
        n_obs = 100
        chunk_size = 10
        start_index = 30  # Aligned with chunk 3

        sampler = SliceSampler(
            n_obs=n_obs,
            batch_size=5,
            chunk_size=chunk_size,
            preload_nchunks=2,
            start_index=start_index,
        )

        all_indices = set()
        for slices, _, _ in sampler:
            for s in slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(start_index, n_obs))
        assert all_indices == expected
        assert min(all_indices) == start_index
        assert max(all_indices) == n_obs - 1

    def test_start_index_not_at_chunk_boundary(self):
        """Test start_index not aligned with chunk boundary."""
        n_obs = 100
        chunk_size = 10
        start_index = 35  # Not aligned - middle of chunk 3

        sampler = SliceSampler(
            n_obs=n_obs,
            batch_size=5,
            chunk_size=chunk_size,
            preload_nchunks=2,
            start_index=start_index,
        )

        all_indices = set()
        for slices, _, _ in sampler:
            for s in slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(start_index, n_obs))
        assert all_indices == expected
        assert min(all_indices) == start_index

    def test_start_index_near_end(self):
        """Test start_index near the end of dataset."""
        n_obs = 100
        chunk_size = 10
        start_index = 90

        sampler = SliceSampler(
            n_obs=n_obs,
            batch_size=3,
            chunk_size=chunk_size,
            preload_nchunks=1,
            start_index=start_index,
        )

        all_indices = set()
        for slices, _, _ in sampler:
            for s in slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(start_index, n_obs))
        assert all_indices == expected
        assert len(all_indices) == 10


class TestSliceSamplerEndIndex:
    """Tests for SliceSampler with custom end_index."""

    def test_end_index_at_chunk_boundary(self):
        """Test end_index aligned with chunk boundary."""
        n_obs = 100
        chunk_size = 10
        end_index = 50  # Aligned with end of chunk 4

        sampler = SliceSampler(
            n_obs=n_obs,
            batch_size=5,
            chunk_size=chunk_size,
            preload_nchunks=2,
            end_index=end_index,
        )

        all_indices = set()
        for slices, _, _ in sampler:
            for s in slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(0, end_index))
        assert all_indices == expected
        assert max(all_indices) == end_index - 1

    def test_end_index_not_at_chunk_boundary(self):
        """Test end_index not aligned with chunk boundary."""
        n_obs = 100
        chunk_size = 10
        end_index = 47  # Middle of chunk 4

        sampler = SliceSampler(
            n_obs=n_obs,
            batch_size=5,
            chunk_size=chunk_size,
            preload_nchunks=2,
            end_index=end_index,
        )

        all_indices = set()
        for slices, _, _ in sampler:
            for s in slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(0, end_index))
        assert all_indices == expected
        assert max(all_indices) == end_index - 1


class TestSliceSamplerBothIndices:
    """Tests for SliceSampler with both start_index and end_index."""

    def test_both_at_chunk_boundaries(self):
        """Test both start and end aligned with chunk boundaries."""
        n_obs = 100
        chunk_size = 10
        start_index = 20
        end_index = 60

        sampler = SliceSampler(
            n_obs=n_obs,
            batch_size=5,
            chunk_size=chunk_size,
            preload_nchunks=2,
            start_index=start_index,
            end_index=end_index,
        )

        all_indices = set()
        for slices, _, _ in sampler:
            for s in slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(start_index, end_index))
        assert all_indices == expected
        assert len(all_indices) == end_index - start_index

    def test_both_not_at_chunk_boundaries(self):
        """Test both start and end not aligned with chunk boundaries."""
        n_obs = 100
        chunk_size = 10
        start_index = 23
        end_index = 67

        sampler = SliceSampler(
            n_obs=n_obs,
            batch_size=5,
            chunk_size=chunk_size,
            preload_nchunks=2,
            start_index=start_index,
            end_index=end_index,
        )

        all_indices = set()
        for slices, _, _ in sampler:
            for s in slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(start_index, end_index))
        assert all_indices == expected
        assert min(all_indices) == start_index
        assert max(all_indices) == end_index - 1

    def test_single_chunk_span(self):
        """Test start and end within a single chunk."""
        n_obs = 100
        chunk_size = 10
        start_index = 22
        end_index = 28  # Same chunk as start

        sampler = SliceSampler(
            n_obs=n_obs,
            batch_size=2,
            chunk_size=chunk_size,
            preload_nchunks=1,
            start_index=start_index,
            end_index=end_index,
        )

        all_indices = set()
        for slices, _, _ in sampler:
            for s in slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(start_index, end_index))
        assert all_indices == expected
        assert len(all_indices) == 6

    def test_worker_shard_simulation(self):
        """Test simulating DataLoader worker sharding (600 obs, 4 workers)."""
        n_obs = 600
        chunk_size = 10
        num_workers = 4
        per_worker = n_obs // num_workers  # 150

        all_worker_indices = set()
        for worker_id in range(num_workers):
            start_index = worker_id * per_worker
            if worker_id == num_workers - 1:
                end_index = n_obs
            else:
                end_index = start_index + per_worker

            sampler = SliceSampler(
                n_obs=n_obs,
                batch_size=10,
                chunk_size=chunk_size,
                preload_nchunks=4,
                start_index=start_index,
                end_index=end_index,
            )

            worker_indices = set()
            for slices, _, _ in sampler:
                for s in slices:
                    worker_indices.update(range(s.start, s.stop))

            # Check this worker got the right range
            expected = set(range(start_index, end_index))
            assert worker_indices == expected

            # Add to global set
            all_worker_indices.update(worker_indices)

        # All workers together should cover the full dataset exactly once
        assert all_worker_indices == set(range(n_obs))


class TestSliceSamplerWithShuffle:
    """Tests for SliceSampler with shuffling enabled."""

    def test_shuffle_with_start_index(self):
        """Test shuffle works correctly with non-zero start_index."""
        n_obs = 100
        chunk_size = 10
        start_index = 30

        sampler = SliceSampler(
            n_obs=n_obs,
            batch_size=5,
            chunk_size=chunk_size,
            preload_nchunks=2,
            start_index=start_index,
            shuffle=True,
            rng=np.random.default_rng(42),
        )

        all_indices = set()
        for slices, _, _ in sampler:
            for s in slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(start_index, n_obs))
        assert all_indices == expected

    def test_shuffle_with_both_indices(self):
        """Test shuffle works correctly with both start and end index."""
        n_obs = 100
        chunk_size = 10
        start_index = 25
        end_index = 75

        sampler = SliceSampler(
            n_obs=n_obs,
            batch_size=5,
            chunk_size=chunk_size,
            preload_nchunks=2,
            start_index=start_index,
            end_index=end_index,
            shuffle=True,
            rng=np.random.default_rng(42),
        )

        all_indices = set()
        for slices, _, _ in sampler:
            for s in slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(start_index, end_index))
        assert all_indices == expected


class TestSliceSamplerEdgeCases:
    """Tests for edge cases."""

    def test_very_small_shard(self):
        """Test with a very small shard (smaller than batch_size)."""
        n_obs = 100
        chunk_size = 10
        start_index = 95
        end_index = 100  # Only 5 observations

        sampler = SliceSampler(
            n_obs=n_obs,
            batch_size=10,  # Larger than shard size
            chunk_size=chunk_size,
            preload_nchunks=1,
            start_index=start_index,
            end_index=end_index,
        )

        all_indices = set()
        for slices, _, _ in sampler:
            for s in slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(start_index, end_index))
        assert all_indices == expected

    def test_start_equals_chunk_size(self):
        """Test start_index exactly equals chunk_size."""
        n_obs = 100
        chunk_size = 10
        start_index = 10  # Exactly one chunk in

        sampler = SliceSampler(
            n_obs=n_obs,
            batch_size=5,
            chunk_size=chunk_size,
            preload_nchunks=2,
            start_index=start_index,
        )

        all_indices = set()
        for slices, _, _ in sampler:
            for s in slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(start_index, n_obs))
        assert all_indices == expected
        assert min(all_indices) == start_index

    @pytest.mark.parametrize(
        "start_index,end_index",
        [
            (0, 100),  # Full range
            (15, 85),  # Both non-aligned
            (20, 80),  # Both aligned
            (0, 50),  # Only end set
            (50, 100),  # Only start set (effectively)
        ],
    )
    def test_parametrized_ranges(self, start_index, end_index):
        """Test various start/end combinations cover correct range."""
        n_obs = 100
        chunk_size = 10

        sampler = SliceSampler(
            n_obs=n_obs,
            batch_size=5,
            chunk_size=chunk_size,
            preload_nchunks=2,
            start_index=start_index,
            end_index=end_index,
        )

        all_indices = set()
        for slices, _, _ in sampler:
            for s in slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(start_index, end_index))
        assert all_indices == expected

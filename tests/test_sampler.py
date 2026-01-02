"""Tests for SliceSampler."""

from __future__ import annotations

import numpy as np
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
            start_index=0,
            end_index=n_obs,
            batch_size=batch_size,
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
        )

        all_indices = set()
        for load_request in sampler:
            for s in load_request.slices:
                all_indices.update(range(s.start, s.stop))

        assert all_indices == set(range(n_obs))

    def test_batch_sizes(self):
        """Test that batch sizes match expected carry-over pattern."""
        n_obs = 100
        chunk_size = 10
        preload_nchunks = 2
        batch_size = 7
        
        # Example with these params (chunk_size=10, preload_nchunks=2, batch_size=7):
        # Each iter loads chunk_size * preload_nchunks = 20 obs
        # Iter 1: 20 obs → [7, 7, 6], leftover=6
        # Iter 2: 20 + 6 = 26 → [7, 7, 7, 5], leftover=5
        # Iter 3: 20 + 5 = 25 → [7, 7, 7, 4], leftover=4
        # Iter 4: 20 + 4 = 24 → [7, 7, 7, 3], leftover=3
        # Iter 5: 20 + 3 = 23 → [7, 7, 7, 2], final partial yielded
        import math

        obs_per_iter = chunk_size * preload_nchunks
        n_iters = math.ceil(n_obs / obs_per_iter)

        expected_sizes_per_iter = []
        leftover = 0
        for i in range(n_iters):
            total_obs = obs_per_iter + leftover
            n_full_batches = total_obs // batch_size
            remainder = total_obs % batch_size
            sizes = [batch_size] * n_full_batches
            if remainder > 0:
                sizes.append(remainder)
            expected_sizes_per_iter.append(sizes)
            leftover = remainder

        sampler = SliceSampler(
            start_index=0,
            end_index=n_obs,
            batch_size=batch_size,
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
        )

        for i, load_request in enumerate(sampler):
            actual_sizes = [len(split) for split in load_request.splits]
            assert actual_sizes == expected_sizes_per_iter[i], (
                f"Iter {i}: expected {expected_sizes_per_iter[i]}, got {actual_sizes}"
            )


class TestSliceSamplerStartIndex:
    """Tests for SliceSampler with non-zero start_index."""

    def test_start_index_at_chunk_boundary(self):
        """Test start_index aligned with chunk boundary."""
        n_obs = 100
        chunk_size = 10
        start_index = 30  # Aligned with chunk 3

        sampler = SliceSampler(
            start_index=start_index,
            end_index=n_obs,
            batch_size=5,
            chunk_size=chunk_size,
            preload_nchunks=2,
        )

        all_indices = set()
        for load_request in sampler:
            for s in load_request.slices:
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
            start_index=start_index,
            end_index=n_obs,
            batch_size=5,
            chunk_size=chunk_size,
            preload_nchunks=2,
        )

        all_indices = set()
        for load_request in sampler:
            for s in load_request.slices:
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
            start_index=start_index,
            end_index=n_obs,
            batch_size=3,
            chunk_size=chunk_size,
            preload_nchunks=1,
        )

        all_indices = set()
        for load_request in sampler:
            for s in load_request.slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(start_index, n_obs))
        assert all_indices == expected
        assert len(all_indices) == 10


class TestSliceSamplerEndIndex:
    """Tests for SliceSampler with custom end_index."""

    def test_end_index_at_chunk_boundary(self):
        """Test end_index aligned with chunk boundary."""
        chunk_size = 10
        end_index = 50  # Aligned with end of chunk 4

        sampler = SliceSampler(
            start_index=0,
            end_index=end_index,
            batch_size=5,
            chunk_size=chunk_size,
            preload_nchunks=2,
        )

        all_indices = set()
        for load_request in sampler:
            for s in load_request.slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(0, end_index))
        assert all_indices == expected
        assert max(all_indices) == end_index - 1

    def test_end_index_not_at_chunk_boundary(self):
        """Test end_index not aligned with chunk boundary."""
        chunk_size = 10
        end_index = 47  # Middle of chunk 4

        sampler = SliceSampler(
            start_index=0,
            end_index=end_index,
            batch_size=5,
            chunk_size=chunk_size,
            preload_nchunks=2,
        )

        all_indices = set()
        for load_request in sampler:
            for s in load_request.slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(0, end_index))
        assert all_indices == expected
        assert max(all_indices) == end_index - 1


class TestSliceSamplerBothIndices:
    """Tests for SliceSampler with both start_index and end_index."""

    def test_both_at_chunk_boundaries(self):
        """Test both start and end aligned with chunk boundaries."""
        chunk_size = 10
        start_index = 20
        end_index = 60

        sampler = SliceSampler(
            start_index=start_index,
            end_index=end_index,
            batch_size=5,
            chunk_size=chunk_size,
            preload_nchunks=2,
        )

        all_indices = set()
        for load_request in sampler:
            for s in load_request.slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(start_index, end_index))
        assert all_indices == expected
        assert len(all_indices) == end_index - start_index

    def test_both_not_at_chunk_boundaries(self):
        """Test both start and end not aligned with chunk boundaries."""
        chunk_size = 10
        start_index = 23
        end_index = 67

        sampler = SliceSampler(
            start_index=start_index,
            end_index=end_index,
            batch_size=5,
            chunk_size=chunk_size,
            preload_nchunks=2,
        )

        all_indices = set()
        for load_request in sampler:
            for s in load_request.slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(start_index, end_index))
        assert all_indices == expected
        assert min(all_indices) == start_index
        assert max(all_indices) == end_index - 1

    def test_single_chunk_span(self):
        """Test start and end within a single chunk."""
        chunk_size = 10
        start_index = 22
        end_index = 28  # Same chunk as start

        sampler = SliceSampler(
            start_index=start_index,
            end_index=end_index,
            batch_size=2,
            chunk_size=chunk_size,
            preload_nchunks=1,
        )

        all_indices = set()
        for load_request in sampler:
            for s in load_request.slices:
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
                start_index=start_index,
                end_index=end_index,
                batch_size=10,
                chunk_size=chunk_size,
                preload_nchunks=4,
            )

            worker_indices = set()
            for load_request in sampler:
                for s in load_request.slices:
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
            start_index=start_index,
            end_index=n_obs,
            batch_size=5,
            chunk_size=chunk_size,
            preload_nchunks=2,
            shuffle=True,
            rng=np.random.default_rng(42),
        )

        all_indices = set()
        for load_request in sampler:
            for s in load_request.slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(start_index, n_obs))
        assert all_indices == expected

    def test_shuffle_with_both_indices(self):
        """Test shuffle works correctly with both start and end index."""
        chunk_size = 10
        start_index = 25
        end_index = 75

        sampler = SliceSampler(
            start_index=start_index,
            end_index=end_index,
            batch_size=5,
            chunk_size=chunk_size,
            preload_nchunks=2,
            shuffle=True,
            rng=np.random.default_rng(42),
        )

        all_indices = set()
        for load_request in sampler:
            for s in load_request.slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(start_index, end_index))
        assert all_indices == expected


class TestSliceSamplerEdgeCases:
    """Tests for edge cases."""

    def test_very_small_shard(self):
        """Test with a very small shard (smaller than batch_size)."""
        chunk_size = 10
        start_index = 95
        end_index = 100  # Only 5 observations

        sampler = SliceSampler(
            start_index=start_index,
            end_index=end_index,
            batch_size=10,  # Larger than shard size
            chunk_size=chunk_size,
            preload_nchunks=1,
        )

        all_indices = set()
        for load_request in sampler:
            for s in load_request.slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(start_index, end_index))
        assert all_indices == expected

    def test_start_equals_chunk_size(self):
        """Test start_index exactly equals chunk_size."""
        n_obs = 100
        chunk_size = 10
        start_index = 10  # Exactly one chunk in

        sampler = SliceSampler(
            start_index=start_index,
            end_index=n_obs,
            batch_size=5,
            chunk_size=chunk_size,
            preload_nchunks=2,
        )

        all_indices = set()
        for load_request in sampler:
            for s in load_request.slices:
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
        chunk_size = 10

        sampler = SliceSampler(
            start_index=start_index,
            end_index=end_index,
            batch_size=5,
            chunk_size=chunk_size,
            preload_nchunks=2,
        )

        all_indices = set()
        for load_request in sampler:
            for s in load_request.slices:
                all_indices.update(range(s.start, s.stop))

        expected = set(range(start_index, end_index))
        assert all_indices == expected


class MockWorkerHandle:
    """Simulates torch worker context for testing without actual DataLoader."""

    def __init__(self, worker_id: int, num_workers: int, seed: int = 42):
        self.worker_id = worker_id
        self.num_workers = num_workers
        self._rng = np.random.default_rng(seed)  # Same seed = consistent shuffle across workers

    def shuffle(self, obj):
        self._rng.shuffle(obj)

    def get_part_for_worker(self, obj: np.ndarray) -> np.ndarray:
        chunks_split = np.array_split(obj, self.num_workers)
        return chunks_split[self.worker_id]


class TestSliceSamplerWithWorkers:
    """Tests for SliceSampler with simulated DataLoader workers."""

    def test_two_workers_divisible_config(self):
        """Test 2 workers with divisible config cover full dataset without overlap."""
        n_obs = 200
        chunk_size = 10
        preload_nchunks = 2
        batch_size = 10  # 10 * 2 = 20, divisible by 10
        num_workers = 2

        all_worker_indices = []
        for worker_id in range(num_workers):
            worker_handle = MockWorkerHandle(worker_id, num_workers)
            sampler = SliceSampler(
                start_index=0,
                end_index=n_obs,
                batch_size=batch_size,
                chunk_size=chunk_size,
                preload_nchunks=preload_nchunks,
            )
            sampler.set_worker_handle(worker_handle)

            worker_indices = set()
            for load_request in sampler:
                for s in load_request.slices:
                    worker_indices.update(range(s.start, s.stop))
            all_worker_indices.append(worker_indices)

        # Workers should have disjoint chunks
        assert all_worker_indices[0].isdisjoint(all_worker_indices[1])
        # Together they cover the full dataset
        assert all_worker_indices[0] | all_worker_indices[1] == set(range(n_obs))

    def test_three_workers_divisible_config(self):
        """Test 3 workers with divisible config (odd worker count)."""
        n_obs = 300
        chunk_size = 10
        preload_nchunks = 3
        batch_size = 10  # 10 * 3 = 30, divisible by 10
        num_workers = 3

        all_worker_indices = []
        for worker_id in range(num_workers):
            worker_handle = MockWorkerHandle(worker_id, num_workers)
            sampler = SliceSampler(
                start_index=0,
                end_index=n_obs,
                batch_size=batch_size,
                chunk_size=chunk_size,
                preload_nchunks=preload_nchunks,
            )
            sampler.set_worker_handle(worker_handle)

            worker_indices = set()
            for load_request in sampler:
                for s in load_request.slices:
                    worker_indices.update(range(s.start, s.stop))
            all_worker_indices.append(worker_indices)

        # All workers should have disjoint chunks
        for i in range(num_workers):
            for j in range(i + 1, num_workers):
                assert all_worker_indices[i].isdisjoint(all_worker_indices[j])
        # Together they cover the full dataset
        combined = set()
        for indices in all_worker_indices:
            combined |= indices
        assert combined == set(range(n_obs))

    def test_workers_drop_last_warns(self):
        """Test that drop_last=True with workers emits warning."""
        worker_handle = MockWorkerHandle(0, 2)

        with pytest.warns(UserWarning, match="multiple workers"):
            sampler = SliceSampler(
                start_index=0,
                end_index=100,
                batch_size=7,  # Non-divisible
                chunk_size=10,
                preload_nchunks=2,
                drop_last=True,
            )
            sampler.set_worker_handle(worker_handle)

    def test_workers_non_divisible_without_drop_last_raises(self):
        """Test that non-divisible config without drop_last raises ValueError."""
        worker_handle = MockWorkerHandle(0, 2)

        with pytest.raises(ValueError, match="divisible by batch_size"):
            sampler = SliceSampler(
                start_index=0,
                end_index=100,
                batch_size=7,  # 10 * 2 = 20, not divisible by 7
                chunk_size=10,
                preload_nchunks=2,
                drop_last=False,
            )
            sampler.set_worker_handle(worker_handle)

    def test_two_workers_drop_last_drops_per_worker(self):
        """Test drop_last=True drops only the final partial batch (intermediate partials are for carry-over)."""
        n_obs = 200
        chunk_size = 10
        preload_nchunks = 2
        batch_size = 7  # Non-divisible: 20 / 7 = 2 full batches + 6 leftover per iter
        num_workers = 2

        for worker_id in range(num_workers):
            worker_handle = MockWorkerHandle(worker_id, num_workers)
            with pytest.warns(UserWarning):
                sampler = SliceSampler(
                    start_index=0,
                    end_index=n_obs,
                    batch_size=batch_size,
                    chunk_size=chunk_size,
                    preload_nchunks=preload_nchunks,
                    drop_last=True,
                )
                sampler.set_worker_handle(worker_handle)

            all_requests = list(sampler)
            # On the final iteration, all splits should be batch_size
            # (the final partial is dropped when drop_last=True)
            if all_requests:
                final_request = all_requests[-1]
                for split in final_request.splits:
                    assert len(split) == batch_size, (
                        f"Worker {worker_id}: final request should have no partial, "
                        f"expected {batch_size}, got {len(split)}"
                    )


# =============================================================================
# RangeGroupSampler Tests
# =============================================================================

from annbatch.sampler import GroupRange, RangeGroupSampler


class TestRangeGroupSamplerBasic:
    """Tests for basic RangeGroupSampler functionality."""

    def test_group_pure_batches(self):
        """Test that each load request only contains slices from one group."""
        groups = [
            GroupRange("A", start=0, end=100, node=0),
            GroupRange("B", start=100, end=200, node=0),
        ]

        sampler = RangeGroupSampler(
            groups=groups,
            rank=0,
            world_size=1,
            batch_size=10,
            chunk_size=10,
            preload_nchunks=2,
            shuffle_groups=False,
            shuffle_within=False,
        )

        # Each individual load request must contain slices from only one group
        for load_request in sampler:
            groups_seen = set()
            for s in load_request.slices:
                if s.start < 100:
                    groups_seen.add("A")
                else:
                    groups_seen.add("B")
            assert len(groups_seen) == 1, f"Load request contains mixed groups: {groups_seen}"

    def test_covers_all_groups(self):
        """Test that sampler covers all observations in assigned groups."""
        groups = [
            GroupRange("A", start=0, end=100, node=0),
            GroupRange("B", start=100, end=200, node=0),
        ]

        sampler = RangeGroupSampler(
            groups=groups,
            rank=0,
            world_size=1,
            batch_size=10,
            chunk_size=10,
            preload_nchunks=2,
            shuffle_groups=False,
            shuffle_within=False,
        )

        all_indices = set()
        for load_request in sampler:
            for s in load_request.slices:
                all_indices.update(range(s.start, s.stop))

        assert all_indices == set(range(200))

    def test_small_group_raises_error(self):
        """Test that groups smaller than preload_size raise an error."""
        groups = [
            GroupRange("small", start=0, end=15, node=0),  # 15 < 10*2 = 20
        ]

        with pytest.raises(ValueError, match="smaller than"):
            RangeGroupSampler(
                groups=groups,
                rank=0,
                world_size=1,
                batch_size=5,
                chunk_size=10,
                preload_nchunks=2,
            )

    def test_non_divisible_preload_raises_error(self):
        """Test that preload_size not divisible by batch_size raises an error."""
        groups = [
            GroupRange("A", start=0, end=100, node=0),
        ]

        with pytest.raises(ValueError, match="divisible by batch_size"):
            RangeGroupSampler(
                groups=groups,
                rank=0,
                world_size=1,
                batch_size=7,        # 20 % 7 != 0
                chunk_size=10,
                preload_nchunks=2,
            )

    def test_all_batches_are_batch_size(self):
        """Test that all yielded full batches are exactly batch_size.
        
        Note: splits[-1] may be partial for carry-over to next iteration.
        Only the final iteration's partial is dropped (drop_last enforced).
        """
        groups = [
            GroupRange("A", start=0, end=100, node=0),
        ]
        batch_size = 10  # 20 % 10 == 0 (preload_size = 10 * 2 = 20)

        sampler = RangeGroupSampler(
            groups=groups,
            rank=0,
            world_size=1,
            batch_size=batch_size,
            chunk_size=10,
            preload_nchunks=2,
            shuffle_groups=False,
            shuffle_within=False,
        )

        for load_request in sampler:
            for split in load_request.splits:
                assert len(split) == batch_size, f"Expected {batch_size}, got {len(split)}"

    def test_interleaved_batches(self):
        """Test that interleave=True alternates batches between groups."""
        groups = [
            GroupRange("A", start=0, end=100, node=0),
            GroupRange("B", start=100, end=200, node=0),
        ]

        sampler = RangeGroupSampler(
            groups=groups,
            rank=0,
            world_size=1,
            batch_size=10,
            chunk_size=10,
            preload_nchunks=2,
            shuffle_groups=False,
            shuffle_within=False,
            interleave=True,
        )

        # Collect first few load requests and check they alternate
        requests = list(sampler)

        # First request should be from A (slices < 100)
        # Second request should be from B (slices >= 100)
        # Third request should be from A again, etc.
        for i, req in enumerate(requests[:4]):  # Check first 4
            for s in req.slices:
                if i % 2 == 0:  # Even indices should be from A
                    assert s.start < 100, f"Request {i} should be from group A, got slice {s}"
                else:  # Odd indices should be from B
                    assert s.start >= 100, f"Request {i} should be from group B, got slice {s}"

    def test_sequential_batches(self):
        """Test that interleave=False exhausts one group before moving to next."""
        groups = [
            GroupRange("A", start=0, end=100, node=0),
            GroupRange("B", start=100, end=200, node=0),
        ]

        sampler = RangeGroupSampler(
            groups=groups,
            rank=0,
            world_size=1,
            batch_size=10,
            chunk_size=10,
            preload_nchunks=2,
            shuffle_groups=False,
            shuffle_within=False,
            interleave=False,  # Sequential mode
        )

        requests = list(sampler)

        # First half should be from A, second half from B
        mid = len(requests) // 2
        for i, req in enumerate(requests):
            for s in req.slices:
                if i < mid:
                    assert s.start < 100, f"Request {i} should be from group A"
                else:
                    assert s.start >= 100, f"Request {i} should be from group B"


class TestRangeGroupSamplerDistributed:
    """Tests for RangeGroupSampler with distributed node assignment."""

    def test_node_filtering(self):
        """Test that each node only sees its assigned groups."""
        groups = [
            GroupRange("A", start=0, end=100, node=0),
            GroupRange("B", start=100, end=200, node=1),
            GroupRange("C", start=200, end=300, node=0),
        ]

        # Node 0 should see A and C
        sampler_0 = RangeGroupSampler(
            groups=groups,
            rank=0,
            world_size=2,
            batch_size=10,
            chunk_size=10,
            preload_nchunks=2,
            shuffle_groups=False,
        )

        indices_0 = set()
        for load_request in sampler_0:
            for s in load_request.slices:
                indices_0.update(range(s.start, s.stop))

        # Node 0 should have A (0-100) and C (200-300)
        assert indices_0 == set(range(0, 100)) | set(range(200, 300))
        assert not indices_0.intersection(set(range(100, 200)))  # No B

        # Node 1 should see only B
        sampler_1 = RangeGroupSampler(
            groups=groups,
            rank=1,
            world_size=2,
            batch_size=10,
            chunk_size=10,
            preload_nchunks=2,
            shuffle_groups=False,
        )

        indices_1 = set()
        for load_request in sampler_1:
            for s in load_request.slices:
                indices_1.update(range(s.start, s.stop))

        assert indices_1 == set(range(100, 200))


class TestRangeGroupSamplerWithWorkers:
    """Tests for RangeGroupSampler with multi-worker support."""

    def test_workers_shard_slices(self):
        """Test that workers get disjoint subsets of slices."""
        groups = [
            GroupRange("A", start=0, end=100, node=0),
        ]
        num_workers = 2

        all_worker_indices = []
        for worker_id in range(num_workers):
            worker_handle = MockWorkerHandle(worker_id, num_workers)
            sampler = RangeGroupSampler(
                groups=groups,
                rank=0,
                world_size=1,
                batch_size=10,
                chunk_size=10,
                preload_nchunks=2,
                shuffle_groups=False,
                shuffle_within=False,
            )
            sampler.set_worker_handle(worker_handle)

            worker_indices = set()
            # No warning since preload_size (20) is divisible by batch_size (10)
            for load_request in sampler:
                for s in load_request.slices:
                    worker_indices.update(range(s.start, s.stop))
            all_worker_indices.append(worker_indices)

        # Workers should have disjoint slices
        assert all_worker_indices[0].isdisjoint(all_worker_indices[1])
        # Together they cover the group
        assert all_worker_indices[0] | all_worker_indices[1] == set(range(100))


class TestRangeGroupSamplerFromObs:
    """Tests for RangeGroupSampler.from_obs factory method."""

    def test_from_obs_detects_ranges(self):
        """Test that from_obs correctly detects group boundaries."""
        import pandas as pd

        obs = pd.DataFrame({
            "cell_type": ["A"] * 50 + ["B"] * 30 + ["C"] * 20,
        })
        node_mapping = {"A": 0, "B": 1, "C": 0}

        sampler = RangeGroupSampler.from_obs(
            obs=obs,
            group_col="cell_type",
            node_mapping=node_mapping,
            rank=0,
            world_size=2,
            batch_size=10,
            chunk_size=10,
            preload_nchunks=2,
        )

        # Node 0 should have A (0-50) and C (80-100)
        indices = set()
        for load_request in sampler:
            for s in load_request.slices:
                indices.update(range(s.start, s.stop))

        assert indices == set(range(0, 50)) | set(range(80, 100))


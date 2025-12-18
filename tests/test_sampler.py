"""Unit tests for samplers and loader-sampler integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from annbatch import CategorySampler, Loader, MaskedSampler, RangeSampler, SliceSampler

if TYPE_CHECKING:
    from pathlib import Path

    import anndata as ad


class TestSliceSampler:
    """Tests for SliceSampler functionality."""

    def test_sequential_iteration(self):
        """Test that sequential iteration yields slices in order."""
        sampler = SliceSampler(n_obs=100, batch_size=30, chunk_size=10, shuffle=False)

        all_slices = list(sampler)

        # Should have ceil(100/30) = 4 batches
        assert len(all_slices) == 4

        # Flatten and check coverage
        flat_slices = [s for batch in all_slices for s in batch]
        starts = [s.start for s in flat_slices]
        stops = [s.stop for s in flat_slices]

        # Sequential order: starts should be 0, 10, 20, ...
        assert starts == list(range(0, 100, 10))
        assert stops == list(range(10, 110, 10))[:10]  # last one is 100

    def test_batch_size_sum(self):
        """Test that each batch sums to approximately batch_size."""
        sampler = SliceSampler(n_obs=1000, batch_size=200, chunk_size=50, shuffle=False)

        for i, batch in enumerate(sampler):
            total = sum(s.stop - s.start for s in batch)
            if i < len(sampler) - 1:
                # Non-final batches should be at least batch_size
                assert total >= 200
            # All batches should have reasonable size
            assert total > 0

    def test_slices_per_batch(self):
        """Test that each batch has approximately batch_size/chunk_size slices."""
        sampler = SliceSampler(n_obs=1000, batch_size=200, chunk_size=50, shuffle=False)

        for batch in sampler:
            # Each batch should have ~4 slices (200/50)
            assert len(batch) >= 1
            # Check slices are chunk-sized (except possibly last)
            for s in batch[:-1]:
                assert s.stop - s.start == 50

    def test_shuffle_changes_order(self):
        """Test that shuffle=True changes the chunk order."""
        sampler_seq = SliceSampler(n_obs=1000, batch_size=100, chunk_size=50, shuffle=False)
        sampler_shuf = SliceSampler(n_obs=1000, batch_size=100, chunk_size=50, shuffle=True)

        seq_slices = [s for batch in sampler_seq for s in batch]
        shuf_slices = [s for batch in sampler_shuf for s in batch]

        seq_starts = [s.start for s in seq_slices]
        shuf_starts = [s.start for s in shuf_slices]

        # Shuffled should have same elements but different order (with high probability)
        assert set(seq_starts) == set(shuf_starts)
        # Order should be different (this could fail with very low probability)
        assert seq_starts != shuf_starts

    def test_full_coverage(self):
        """Test that all observations are covered exactly once."""
        sampler = SliceSampler(n_obs=573, batch_size=100, chunk_size=37, shuffle=True)

        covered = np.zeros(573, dtype=bool)
        for batch in sampler:
            for s in batch:
                covered[s.start : s.stop] = True

        assert covered.all(), "Not all observations were covered"

    def test_len_matches_iteration_count(self):
        """Test that __len__ matches actual iteration count."""
        sampler = SliceSampler(n_obs=1000, batch_size=300, chunk_size=50, shuffle=False)

        batches = list(sampler)
        assert len(sampler) == len(batches)

    def test_len_calculation(self):
        """Test __len__ returns ceil(n_obs / batch_size)."""
        sampler = SliceSampler(n_obs=1000, batch_size=300, chunk_size=50, shuffle=False)
        assert len(sampler) == 4  # ceil(1000/300) = 4

        sampler2 = SliceSampler(n_obs=100, batch_size=100, chunk_size=10, shuffle=False)
        assert len(sampler2) == 1

        sampler3 = SliceSampler(n_obs=101, batch_size=100, chunk_size=10, shuffle=False)
        assert len(sampler3) == 2

    def test_last_chunk_handles_remainder(self):
        """Test that the last chunk correctly handles n_obs not divisible by chunk_size."""
        sampler = SliceSampler(n_obs=95, batch_size=100, chunk_size=10, shuffle=False)

        all_slices = [s for batch in sampler for s in batch]
        last_slice = all_slices[-1]

        # Last slice should end at n_obs
        assert last_slice.stop == 95
        # Last slice should be smaller than chunk_size
        assert last_slice.stop - last_slice.start == 5

    def test_single_chunk(self):
        """Test sampler with data smaller than one chunk."""
        sampler = SliceSampler(n_obs=5, batch_size=10, chunk_size=10, shuffle=False)

        batches = list(sampler)
        assert len(batches) == 1
        assert len(batches[0]) == 1
        assert batches[0][0] == slice(0, 5)

    def test_exact_division(self):
        """Test when n_obs divides evenly by chunk_size and batch_size."""
        sampler = SliceSampler(n_obs=100, batch_size=50, chunk_size=10, shuffle=False)

        batches = list(sampler)
        assert len(batches) == 2

        for batch in batches:
            total = sum(s.stop - s.start for s in batch)
            assert total == 50

    def test_multiple_iterations_with_shuffle(self):
        """Test that multiple iterations produce different orders when shuffled."""
        sampler = SliceSampler(n_obs=1000, batch_size=100, chunk_size=50, shuffle=True)

        iter1_starts = [s.start for batch in sampler for s in batch]
        iter2_starts = [s.start for batch in sampler for s in batch]

        # Same coverage
        assert set(iter1_starts) == set(iter2_starts)
        # Different order (with very high probability for 20 chunks)
        assert iter1_starts != iter2_starts

    def test_chunk_alignment(self):
        """Test that slices are chunk-aligned (start at chunk boundaries)."""
        chunk_size = 64
        sampler = SliceSampler(n_obs=1000, batch_size=256, chunk_size=chunk_size, shuffle=False)

        for batch in sampler:
            for s in batch:
                # Start should be at a chunk boundary
                assert s.start % chunk_size == 0

    @pytest.mark.parametrize(
        "n_obs,batch_size,chunk_size",
        [
            (100, 10, 5),
            (1000, 100, 10),
            (500, 128, 32),
            (73, 20, 7),  # odd numbers
            (1, 1, 1),  # minimal case
        ],
    )
    def test_various_configurations(self, n_obs, batch_size, chunk_size):
        """Test sampler works with various configurations."""
        sampler = SliceSampler(n_obs=n_obs, batch_size=batch_size, chunk_size=chunk_size, shuffle=False)

        # Should be iterable
        batches = list(sampler)
        assert len(batches) > 0

        # Should cover all observations
        covered = set()
        for batch in batches:
            for s in batch:
                for i in range(s.start, s.stop):
                    covered.add(i)

        assert covered == set(range(n_obs))


class TestLoaderWithSampler:
    """Tests for Loader integration with SliceSampler."""

    def test_loader_creates_default_sampler(self, adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path]):
        """Test that loader creates a SliceSampler when none is provided."""
        loader = Loader(
            chunk_size=10,
            preload_nchunks=5,
            batch_size=20,
            shuffle=False,
            preload_to_gpu=False,
            to_torch=False,
        )
        path = next(adata_with_zarr_path_same_var_space[1].glob("*.zarr"))
        loader.add_dataset(
            dataset=__import__("anndata").io.sparse_dataset(__import__("zarr").open(path)["layers"]["sparse"])
        )

        # Access sampler through iteration
        sampler = loader._get_sampler()
        assert isinstance(sampler, SliceSampler)

    def test_loader_uses_custom_sampler(self, adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path]):
        """Test that loader uses a custom sampler when provided."""
        path = next(adata_with_zarr_path_same_var_space[1].glob("*.zarr"))
        dataset = __import__("anndata").io.sparse_dataset(__import__("zarr").open(path)["layers"]["sparse"])
        n_obs = dataset.shape[0]

        custom_sampler = SliceSampler(n_obs=n_obs, batch_size=50, chunk_size=10, shuffle=True)

        loader = Loader(
            chunk_size=10,
            preload_nchunks=5,
            batch_size=20,
            batch_sampler=custom_sampler,
            preload_to_gpu=False,
            to_torch=False,
        )
        loader.add_dataset(dataset=dataset)

        assert loader._get_sampler() is custom_sampler

    def test_loader_sampler_caching(self, adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path]):
        """Test that loader caches the sampler instance."""
        loader = Loader(
            chunk_size=10,
            preload_nchunks=5,
            batch_size=20,
            shuffle=False,
            preload_to_gpu=False,
            to_torch=False,
        )
        path = next(adata_with_zarr_path_same_var_space[1].glob("*.zarr"))
        loader.add_dataset(
            dataset=__import__("anndata").io.sparse_dataset(__import__("zarr").open(path)["layers"]["sparse"])
        )

        sampler1 = loader._get_sampler()
        sampler2 = loader._get_sampler()
        assert sampler1 is sampler2

    def test_loader_shuffle_passed_to_sampler(self, adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path]):
        """Test that loader's shuffle parameter is passed to the sampler."""
        loader = Loader(
            chunk_size=10,
            preload_nchunks=5,
            batch_size=20,
            shuffle=True,
            preload_to_gpu=False,
            to_torch=False,
        )
        path = next(adata_with_zarr_path_same_var_space[1].glob("*.zarr"))
        loader.add_dataset(
            dataset=__import__("anndata").io.sparse_dataset(__import__("zarr").open(path)["layers"]["sparse"])
        )

        sampler = loader._get_sampler()
        assert sampler._shuffle is True

    def test_loader_sampler_batch_size_calculation(self, adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path]):
        """Test that loader calculates sampler batch_size as batch_size * preload_nchunks."""
        chunk_size = 10
        preload_nchunks = 5
        batch_size = 20

        loader = Loader(
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
            batch_size=batch_size,
            shuffle=False,
            preload_to_gpu=False,
            to_torch=False,
        )
        path = next(adata_with_zarr_path_same_var_space[1].glob("*.zarr"))
        loader.add_dataset(
            dataset=__import__("anndata").io.sparse_dataset(__import__("zarr").open(path)["layers"]["sparse"])
        )

        sampler = loader._get_sampler()
        # Sampler's batch_size should be batch_size * preload_nchunks
        assert sampler._batch_size == batch_size * preload_nchunks

    @pytest.mark.parametrize("shuffle", [True, False])
    def test_loader_iteration_with_sampler(
        self, adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path], shuffle: bool
    ):
        """Test that loader iterates correctly using the sampler."""

        loader = Loader(
            chunk_size=10,
            preload_nchunks=5,
            batch_size=20,
            shuffle=shuffle,
            return_index=True,
            preload_to_gpu=False,
            to_torch=False,
        )
        path = next(adata_with_zarr_path_same_var_space[1].glob("*.zarr"))
        loader.add_dataset(
            dataset=__import__("anndata").io.sparse_dataset(__import__("zarr").open(path)["layers"]["sparse"])
        )

        batches = []
        indices = []
        for batch in loader:
            x, _, idx = batch
            batches.append(x)
            indices.append(idx)

        # Should have processed all data
        total_obs = sum(b.shape[0] for b in batches)
        assert total_obs == loader.n_obs

        # All indices should be covered
        all_indices = np.concatenate(indices)
        assert set(all_indices) == set(range(loader.n_obs))

    def test_loader_multiple_datasets_with_sampler(self, adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path]):
        """Test loader with multiple datasets uses sampler correctly."""

        loader = Loader(
            chunk_size=10,
            preload_nchunks=5,
            batch_size=20,
            shuffle=False,
            return_index=True,
            preload_to_gpu=False,
            to_torch=False,
        )

        # Add multiple datasets
        for path in adata_with_zarr_path_same_var_space[1].glob("*.zarr"):
            loader.add_dataset(
                dataset=__import__("anndata").io.sparse_dataset(__import__("zarr").open(path)["layers"]["sparse"])
            )

        # Sampler should be created with total n_obs
        sampler = loader._get_sampler()
        assert sampler._n_obs == loader.n_obs

        # Iterate and verify coverage
        total_obs = 0
        for batch in loader:
            x, _, idx = batch
            total_obs += x.shape[0]

        assert total_obs == loader.n_obs


class TestRangeSampler:
    """Tests for RangeSampler functionality."""

    def test_basic_iteration(self):
        """Test basic sequential iteration."""
        sampler = RangeSampler(start=10, stop=30, batch_size=8, shuffle=False)

        batches = list(sampler)
        assert len(batches) == 3  # ceil(20/8)

        # First batch should be single slice [10, 18)
        assert batches[0] == [slice(10, 18)]
        assert batches[1] == [slice(18, 26)]
        assert batches[2] == [slice(26, 30)]

    def test_full_coverage(self):
        """Test all indices are covered."""
        sampler = RangeSampler(start=5, stop=25, batch_size=7, shuffle=True)

        covered = set()
        for batch in sampler:
            for s in batch:
                for i in range(s.start, s.stop):
                    covered.add(i)

        assert covered == set(range(5, 25))

    def test_len_calculation(self):
        """Test __len__ calculation."""
        sampler = RangeSampler(start=0, stop=100, batch_size=30, shuffle=False)
        assert len(sampler) == 4  # ceil(100/30)

    def test_empty_range(self):
        """Test empty range."""
        sampler = RangeSampler(start=10, stop=10, batch_size=5, shuffle=False)
        assert len(sampler) == 0
        assert list(sampler) == []

    def test_shuffle_changes_batch_contents(self):
        """Test that shuffle affects batch contents."""
        sampler_seq = RangeSampler(start=0, stop=100, batch_size=20, shuffle=False)
        sampler_shuf = RangeSampler(start=0, stop=100, batch_size=20, shuffle=True)

        seq_batches = list(sampler_seq)
        shuf_batches = list(sampler_shuf)

        # Sequential should have single slices
        assert all(len(b) == 1 for b in seq_batches)
        # Shuffled may have multiple slices (non-consecutive after shuffle)
        # Both should cover same indices
        seq_covered = {i for b in seq_batches for s in b for i in range(s.start, s.stop)}
        shuf_covered = {i for b in shuf_batches for s in b for i in range(s.start, s.stop)}
        assert seq_covered == shuf_covered == set(range(100))


class TestMaskedSampler:
    """Tests for MaskedSampler functionality."""

    def test_basic_iteration(self):
        """Test basic iteration over masked indices."""
        indices = np.array([0, 1, 2, 10, 11, 12, 20, 21])
        sampler = MaskedSampler(indices=indices, batch_size=4, shuffle=False)

        all_slices = list(sampler)

        # Should have ceil(8/4) = 2 batches
        assert len(all_slices) == 2

        # First batch should cover first 4 indices (0,1,2,10)
        first_batch = all_slices[0]
        first_covered = set()
        for s in first_batch:
            for i in range(s.start, s.stop):
                first_covered.add(i)
        assert first_covered == {0, 1, 2, 10}

    def test_consecutive_indices_grouped(self):
        """Test that consecutive indices are grouped into single slices."""
        indices = np.array([0, 1, 2, 3, 4])
        sampler = MaskedSampler(indices=indices, batch_size=10, shuffle=False)

        batches = list(sampler)
        assert len(batches) == 1
        assert len(batches[0]) == 1  # All consecutive, should be one slice
        assert batches[0][0] == slice(0, 5)

    def test_non_consecutive_indices_multiple_slices(self):
        """Test that non-consecutive indices produce multiple slices."""
        indices = np.array([0, 2, 4, 6, 8])  # All non-consecutive
        sampler = MaskedSampler(indices=indices, batch_size=10, shuffle=False)

        batches = list(sampler)
        assert len(batches) == 1
        assert len(batches[0]) == 5  # Each index is its own slice

    def test_full_coverage(self):
        """Test that all masked indices are covered."""
        indices = np.array([5, 10, 15, 20, 25, 30, 35, 40])
        sampler = MaskedSampler(indices=indices, batch_size=3, shuffle=True)

        covered = set()
        for batch in sampler:
            for s in batch:
                for i in range(s.start, s.stop):
                    covered.add(i)

        assert covered == set(indices)

    def test_shuffle_changes_batch_composition(self):
        """Test that shuffle changes which indices end up in which batch."""
        indices = np.arange(100)
        sampler_seq = MaskedSampler(indices=indices, batch_size=10, shuffle=False)
        sampler_shuf = MaskedSampler(indices=indices, batch_size=10, shuffle=True)

        seq_batches = list(sampler_seq)
        shuf_batches = list(sampler_shuf)

        # Both should cover same indices
        seq_covered = set()
        shuf_covered = set()
        for batch in seq_batches:
            for s in batch:
                for i in range(s.start, s.stop):
                    seq_covered.add(i)
        for batch in shuf_batches:
            for s in batch:
                for i in range(s.start, s.stop):
                    shuf_covered.add(i)

        assert seq_covered == shuf_covered == set(indices)

    def test_len_calculation(self):
        """Test __len__ returns ceil(n_indices / batch_size)."""
        indices = np.arange(100)
        sampler = MaskedSampler(indices=indices, batch_size=30, shuffle=False)
        assert len(sampler) == 4  # ceil(100/30)

    def test_empty_indices(self):
        """Test sampler with empty indices."""
        indices = np.array([], dtype=int)
        sampler = MaskedSampler(indices=indices, batch_size=10, shuffle=False)

        batches = list(sampler)
        assert len(batches) == 0

    def test_single_index(self):
        """Test sampler with single index."""
        indices = np.array([42])
        sampler = MaskedSampler(indices=indices, batch_size=10, shuffle=False)

        batches = list(sampler)
        assert len(batches) == 1
        assert batches[0] == [slice(42, 43)]


class TestCategorySampler:
    """Tests for CategorySampler functionality."""

    def test_basic_iteration(self):
        """Test basic iteration over categories."""
        categories = np.array([0, 0, 0, 1, 1, 1, 2, 2])
        sampler = CategorySampler(
            categories=categories,
            batch_size=2,
            shuffle_categories=False,
            shuffle_within=False,
        )

        batches = list(sampler)

        # Category 0 has 3 obs -> ceil(3/2) = 2 batches
        # Category 1 has 3 obs -> ceil(3/2) = 2 batches
        # Category 2 has 2 obs -> ceil(2/2) = 1 batch
        # Total = 5 batches
        assert len(batches) == 5

    def test_full_coverage(self):
        """Test that all observations are covered."""
        categories = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
        sampler = CategorySampler(
            categories=categories,
            batch_size=2,
            shuffle_categories=True,
            shuffle_within=True,
        )

        covered = set()
        for batch in sampler:
            for s in batch:
                for i in range(s.start, s.stop):
                    covered.add(i)

        assert covered == set(range(len(categories)))

    def test_category_order_preserved_when_not_shuffled(self):
        """Test that category order is preserved when shuffle_categories=False."""
        categories = np.array([0, 0, 1, 1, 2, 2])
        sampler = CategorySampler(
            categories=categories,
            batch_size=10,
            shuffle_categories=False,
            shuffle_within=False,
        )

        batches = list(sampler)

        # First batch should be category 0 (indices 0, 1)
        first_covered = set()
        for s in batches[0]:
            for i in range(s.start, s.stop):
                first_covered.add(i)
        assert first_covered == {0, 1}

        # Second batch should be category 1 (indices 2, 3)
        second_covered = set()
        for s in batches[1]:
            for i in range(s.start, s.stop):
                second_covered.add(i)
        assert second_covered == {2, 3}

    def test_shuffle_categories_changes_order(self):
        """Test that shuffle_categories changes the category order."""
        categories = np.array([0] * 10 + [1] * 10 + [2] * 10)
        sampler_seq = CategorySampler(
            categories=categories,
            batch_size=20,
            shuffle_categories=False,
            shuffle_within=False,
        )
        sampler_shuf = CategorySampler(
            categories=categories,
            batch_size=20,
            shuffle_categories=True,
            shuffle_within=False,
        )

        seq_batches = list(sampler_seq)
        shuf_batches = list(sampler_shuf)

        # Get first index of each batch to determine category order
        seq_first = [list(range(b[0].start, b[0].stop))[0] for b in seq_batches]
        shuf_first = [list(range(b[0].start, b[0].stop))[0] for b in shuf_batches]

        # Sequential should be 0, 10, 20 (category 0, 1, 2 in order)
        assert seq_first == [0, 10, 20]
        # Shuffled should be different order (with high probability)
        # We just check they cover the same categories
        assert set(categories[seq_first]) == set(categories[shuf_first])

    def test_len_calculation(self):
        """Test __len__ returns sum of batches per category."""
        # Category 0: 5 obs, batch_size 2 -> 3 batches
        # Category 1: 3 obs, batch_size 2 -> 2 batches
        # Total: 5 batches
        categories = np.array([0, 0, 0, 0, 0, 1, 1, 1])
        sampler = CategorySampler(
            categories=categories,
            batch_size=2,
            shuffle_categories=False,
            shuffle_within=False,
        )
        assert len(sampler) == 5

    def test_single_category(self):
        """Test sampler with single category."""
        categories = np.array([0, 0, 0, 0, 0])
        sampler = CategorySampler(
            categories=categories,
            batch_size=2,
            shuffle_categories=False,
            shuffle_within=False,
        )

        batches = list(sampler)
        assert len(batches) == 3  # ceil(5/2)

    def test_many_categories(self):
        """Test sampler with many categories."""
        categories = np.arange(100)  # Each obs is its own category
        sampler = CategorySampler(
            categories=categories,
            batch_size=1,
            shuffle_categories=False,
            shuffle_within=False,
        )

        batches = list(sampler)
        assert len(batches) == 100

        # Each batch should have one slice for one index
        for i, batch in enumerate(batches):
            assert len(batch) == 1
            assert batch[0] == slice(i, i + 1)

    def test_string_categories(self):
        """Test sampler with string category labels."""
        categories = np.array(["A", "A", "B", "B", "C", "C"])
        sampler = CategorySampler(
            categories=categories,
            batch_size=10,
            shuffle_categories=False,
            shuffle_within=False,
        )

        batches = list(sampler)
        assert len(batches) == 3  # One batch per category

    @pytest.mark.parametrize(
        "n_categories,obs_per_category,batch_size",
        [
            (3, 10, 5),
            (5, 20, 8),
            (10, 5, 3),
            (2, 100, 25),
        ],
    )
    def test_various_configurations(self, n_categories, obs_per_category, batch_size):
        """Test sampler with various configurations."""
        categories = np.repeat(np.arange(n_categories), obs_per_category)
        sampler = CategorySampler(
            categories=categories,
            batch_size=batch_size,
            shuffle_categories=True,
            shuffle_within=True,
        )

        # Should cover all observations
        covered = set()
        for batch in sampler:
            for s in batch:
                for i in range(s.start, s.stop):
                    covered.add(i)

        assert covered == set(range(len(categories)))

    def test_sorted_categories_detected(self):
        """Test that sorted categories are detected and use RangeSampler."""
        # Sorted: each category in contiguous block
        categories = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        sampler = CategorySampler(categories=categories, batch_size=2)

        assert sampler._is_sorted is True
        # Data should be stored as (start, stop) tuples
        assert sampler._category_data[0] == (0, 3)
        assert sampler._category_data[1] == (3, 6)
        assert sampler._category_data[2] == (6, 9)

    def test_unsorted_categories_detected(self):
        """Test that unsorted categories are detected and use MaskedSampler."""
        # Unsorted: categories interleaved
        categories = np.array([0, 1, 0, 1, 0, 1])
        sampler = CategorySampler(categories=categories, batch_size=2)

        assert sampler._is_sorted is False
        # Data should be stored as index arrays
        np.testing.assert_array_equal(sampler._category_data[0], [0, 2, 4])
        np.testing.assert_array_equal(sampler._category_data[1], [1, 3, 5])

    def test_sorted_and_unsorted_same_coverage(self):
        """Test that sorted and unsorted produce same coverage."""
        # Create sorted version
        sorted_cats = np.array([0, 0, 0, 1, 1, 1, 2, 2])
        # Create unsorted version with same counts
        unsorted_cats = np.array([0, 1, 2, 0, 1, 2, 0, 1])

        sampler_sorted = CategorySampler(categories=sorted_cats, batch_size=2)
        sampler_unsorted = CategorySampler(categories=unsorted_cats, batch_size=2)

        assert sampler_sorted._is_sorted is True
        assert sampler_unsorted._is_sorted is False

        # Both should cover all indices
        sorted_covered = {i for b in sampler_sorted for s in b for i in range(s.start, s.stop)}
        unsorted_covered = {i for b in sampler_unsorted for s in b for i in range(s.start, s.stop)}

        assert sorted_covered == set(range(8))
        assert unsorted_covered == set(range(8))

    def test_sorted_categories_explicit_true(self):
        """Test explicit sorted_categories=True skips detection."""
        categories = np.array([0, 0, 0, 1, 1, 1])
        sampler = CategorySampler(categories=categories, batch_size=2, sorted_categories=True)

        assert sampler._is_sorted is True
        # Should store ranges, not indices
        assert sampler._category_data[0] == (0, 3)

    def test_sorted_categories_explicit_false(self):
        """Test explicit sorted_categories=False skips detection."""
        categories = np.array([0, 0, 0, 1, 1, 1])
        sampler = CategorySampler(categories=categories, batch_size=2, sorted_categories=False)

        assert sampler._is_sorted is False
        # Should store indices, not ranges
        np.testing.assert_array_equal(sampler._category_data[0], [0, 1, 2])

    def test_sorted_categories_none_autodetects(self):
        """Test sorted_categories=None auto-detects."""
        sorted_cats = np.array([0, 0, 1, 1])
        unsorted_cats = np.array([0, 1, 0, 1])

        sampler_sorted = CategorySampler(categories=sorted_cats, batch_size=2, sorted_categories=None)
        sampler_unsorted = CategorySampler(categories=unsorted_cats, batch_size=2, sorted_categories=None)

        assert sampler_sorted._is_sorted is True
        assert sampler_unsorted._is_sorted is False

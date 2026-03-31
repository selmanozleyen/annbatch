"""Tests for GroupedCollection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import zarr

from annbatch import CategoricalSampler, GroupedCollection, Loader, write_sharded
from annbatch.io import V1_GROUPED_ENCODING

if TYPE_CHECKING:
    from pathlib import Path


def test_grouped_collection_basic(adata_with_h5_path_different_var_space: tuple[ad.AnnData, Path], tmp_path: Path):
    paths = sorted(p for p in adata_with_h5_path_different_var_space[1].iterdir() if p.suffix == ".h5ad")[:3]
    output = tmp_path / "grouped.zarr"
    collection = GroupedCollection(output).add_adatas(
        paths,
        groupby="label",
        n_obs_per_chunk=2,
        zarr_shard_size=10,
        n_obs_per_dataset=200,
        shuffle=False,
    )
    assert not collection.is_empty
    store = zarr.open(output)
    assert V1_GROUPED_ENCODING.items() <= store.attrs.items()
    assert list(store.attrs["groupby_keys"]) == ["label"]


def test_grouped_collection_compound_groupby(
    adata_with_h5_path_different_var_space: tuple[ad.AnnData, Path], tmp_path: Path
):
    paths = sorted(p for p in adata_with_h5_path_different_var_space[1].iterdir() if p.suffix == ".h5ad")[:3]
    output = tmp_path / "grouped_compound.zarr"
    collection = GroupedCollection(output).add_adatas(
        paths,
        groupby=["label", "store_id"],
        n_obs_per_chunk=2,
        zarr_shard_size=10,
        n_obs_per_dataset=200,
        shuffle=True,
        random_seed=0,
    )
    assert not collection.is_empty
    store = zarr.open(output)
    assert list(store.attrs["groupby_keys"]) == ["label", "store_id"]


def test_group_index_metadata(adata_with_h5_path_different_var_space: tuple[ad.AnnData, Path], tmp_path: Path):
    paths = sorted(p for p in adata_with_h5_path_different_var_space[1].iterdir() if p.suffix == ".h5ad")[:3]
    output = tmp_path / "grouped_meta.zarr"
    collection = GroupedCollection(output).add_adatas(
        paths,
        groupby=["label", "store_id"],
        n_obs_per_chunk=2,
        zarr_shard_size=10,
        n_obs_per_dataset=200,
        random_seed=0,
    )
    group_index = collection.group_index
    assert {"label", "store_id", "start", "stop", "count"}.issubset(group_index.columns)
    n_obs_total = sum(ad.read_h5ad(p).n_obs for p in paths)
    assert int(group_index["count"].sum()) == n_obs_total
    assert int(group_index["start"].iloc[0]) == 0
    assert np.all(group_index["start"].to_numpy() < group_index["stop"].to_numpy())
    assert np.array_equal(
        group_index["count"].to_numpy(),
        group_index["stop"].to_numpy() - group_index["start"].to_numpy(),
    )


def test_grouped_data_is_contiguous_by_group(
    adata_with_h5_path_different_var_space: tuple[ad.AnnData, Path], tmp_path: Path
):
    paths = sorted(p for p in adata_with_h5_path_different_var_space[1].iterdir() if p.suffix == ".h5ad")[:3]
    output = tmp_path / "grouped_contiguous.zarr"
    collection = GroupedCollection(output).add_adatas(
        paths,
        groupby=["label", "store_id"],
        n_obs_per_chunk=2,
        zarr_shard_size=10,
        n_obs_per_dataset=200,
        shuffle=False,
    )
    adata_grouped = ad.concat([ad.io.read_elem(g) for g in collection], join="outer")
    grouped_obs = adata_grouped.obs[["label", "store_id"]].astype("string").fillna("<NA>")

    group_index = collection.group_index
    for row in group_index.itertuples():
        sliced = grouped_obs.iloc[int(row.start) : int(row.stop)]
        assert (sliced["label"] == str(row.label)).all()
        assert (sliced["store_id"] == str(row.store_id)).all()


def test_dataset_groupby_single_column(
    adata_with_h5_path_different_var_space: tuple[ad.AnnData, Path], tmp_path: Path
):
    """dataset_groupby='label' produces one dataset per unique label value."""
    paths = sorted(p for p in adata_with_h5_path_different_var_space[1].iterdir() if p.suffix == ".h5ad")[:3]
    output = tmp_path / "grouped_ds_by_label.zarr"
    collection = GroupedCollection(output).add_adatas(
        paths,
        groupby=["label", "store_id"],
        dataset_groupby="label",
        n_obs_per_chunk=2,
        zarr_shard_size=10,
        shuffle=False,
    )
    assert not collection.is_empty

    concat_adata = ad.concat([ad.read_h5ad(p) for p in paths], join="outer")
    n_unique_labels = concat_adata.obs["label"].nunique()
    assert len(collection._dataset_keys) == n_unique_labels

    adata_grouped = ad.concat([ad.io.read_elem(g) for g in collection], join="outer")
    n_obs_total = sum(ad.read_h5ad(p).n_obs for p in paths)
    assert adata_grouped.n_obs == n_obs_total

    group_index = collection.group_index
    assert int(group_index["count"].sum()) == n_obs_total
    for row in group_index.itertuples():
        sliced = adata_grouped.obs.iloc[int(row.start) : int(row.stop)]
        assert (sliced["label"].astype(str) == str(row.label)).all()
        assert (sliced["store_id"].astype(str) == str(row.store_id)).all()


def test_dataset_groupby_compound(
    adata_with_h5_path_different_var_space: tuple[ad.AnnData, Path], tmp_path: Path
):
    """dataset_groupby=['label', 'store_id'] produces one dataset per (label, store_id) combo."""
    paths = sorted(p for p in adata_with_h5_path_different_var_space[1].iterdir() if p.suffix == ".h5ad")[:3]
    output = tmp_path / "grouped_ds_compound.zarr"
    collection = GroupedCollection(output).add_adatas(
        paths,
        groupby=["label", "store_id"],
        dataset_groupby=["label", "store_id"],
        n_obs_per_chunk=2,
        zarr_shard_size=10,
        shuffle=False,
    )
    concat_adata = ad.concat([ad.read_h5ad(p) for p in paths], join="outer")
    n_unique_combos = concat_adata.obs.groupby(["label", "store_id"], observed=True).ngroups
    assert len(collection._dataset_keys) == n_unique_combos

    group_index = collection.group_index
    n_obs_total = sum(ad.read_h5ad(p).n_obs for p in paths)
    assert int(group_index["count"].sum()) == n_obs_total


def test_dataset_groupby_equals_groupby(
    adata_with_h5_path_different_var_space: tuple[ad.AnnData, Path], tmp_path: Path
):
    """dataset_groupby equal to groupby gives one dataset per group."""
    paths = sorted(p for p in adata_with_h5_path_different_var_space[1].iterdir() if p.suffix == ".h5ad")[:3]
    output = tmp_path / "grouped_ds_eq.zarr"
    collection = GroupedCollection(output).add_adatas(
        paths,
        groupby="label",
        dataset_groupby="label",
        n_obs_per_chunk=2,
        zarr_shard_size=10,
        shuffle=False,
    )
    group_index = collection.group_index
    assert len(collection._dataset_keys) == len(group_index)

    adata_grouped = ad.concat([ad.io.read_elem(g) for g in collection], join="outer")
    for row in group_index.itertuples():
        sliced = adata_grouped.obs.iloc[int(row.start) : int(row.stop)]
        assert len(sliced["label"].unique()) == 1


def test_dataset_groupby_bad_subset_raises(
    adata_with_h5_path_different_var_space: tuple[ad.AnnData, Path], tmp_path: Path
):
    paths = sorted(p for p in adata_with_h5_path_different_var_space[1].iterdir() if p.suffix == ".h5ad")[:3]
    output = tmp_path / "grouped_ds_bad.zarr"
    with pytest.raises(ValueError, match="subset"):
        GroupedCollection(output).add_adatas(
            paths,
            groupby="label",
            dataset_groupby="nonexistent",
            n_obs_per_chunk=2,
            zarr_shard_size=10,
        )


def test_dataset_groupby_bad_prefix_raises(
    adata_with_h5_path_different_var_space: tuple[ad.AnnData, Path], tmp_path: Path
):
    paths = sorted(p for p in adata_with_h5_path_different_var_space[1].iterdir() if p.suffix == ".h5ad")[:3]
    output = tmp_path / "grouped_ds_bad_prefix.zarr"
    with pytest.raises(ValueError, match="prefix"):
        GroupedCollection(output).add_adatas(
            paths,
            groupby=["label", "store_id"],
            dataset_groupby="store_id",
            n_obs_per_chunk=2,
            zarr_shard_size=10,
        )


def test_dataset_groupby_with_categorical_sampler(
    adata_with_h5_path_different_var_space: tuple[ad.AnnData, Path], tmp_path: Path
):
    """CategoricalSampler.from_collection works with dataset_groupby collections."""
    paths = sorted(p for p in adata_with_h5_path_different_var_space[1].iterdir() if p.suffix == ".h5ad")[:3]
    grouped = GroupedCollection(tmp_path / "grouped_ds_catsampler.zarr").add_adatas(
        paths,
        groupby="label",
        dataset_groupby="label",
        n_obs_per_chunk=2,
        zarr_shard_size=10,
        shuffle=False,
    )
    group_index = grouped.group_index
    min_cat_size = int(group_index["count"].min())
    chunk_size = min(min_cat_size, 2)
    batch_size = chunk_size
    sampler = CategoricalSampler.from_collection(
        grouped,
        chunk_size=chunk_size,
        preload_nchunks=2,
        batch_size=batch_size,
        num_samples=20 * batch_size,
        rng=np.random.default_rng(0),
    )
    assert sampler.n_categories == len(group_index)

    loader = Loader(
        batch_sampler=sampler,
        return_index=False,
        preload_to_gpu=False,
        to_torch=False,
    ).use_collection(grouped)

    batch_count = 0
    for batch in loader:
        obs = batch["obs"]
        assert obs is not None
        labels_in_batch = obs["label"].unique()
        assert len(labels_in_batch) == 1
        batch_count += 1
    assert batch_count == 20


def test_grouped_collection_non_empty_raises(
    adata_with_h5_path_different_var_space: tuple[ad.AnnData, Path], tmp_path: Path
):
    paths = sorted(p for p in adata_with_h5_path_different_var_space[1].iterdir() if p.suffix == ".h5ad")[:3]
    output = tmp_path / "grouped_nonempty.zarr"
    collection = GroupedCollection(output).add_adatas(
        paths, groupby="label", n_obs_per_chunk=2, zarr_shard_size=10, n_obs_per_dataset=200
    )
    with pytest.raises(RuntimeError, match="non-empty"):
        collection.add_adatas(paths, groupby="label", n_obs_per_chunk=2, zarr_shard_size=10, n_obs_per_dataset=200)


def test_grouped_collection_missing_groupby_raises(
    adata_with_h5_path_different_var_space: tuple[ad.AnnData, Path], tmp_path: Path
):
    paths = sorted(p for p in adata_with_h5_path_different_var_space[1].iterdir() if p.suffix == ".h5ad")[:3]
    output = tmp_path / "grouped_bad_key.zarr"
    with pytest.raises(ValueError, match="Could not find groupby"):
        GroupedCollection(output).add_adatas(
            paths, groupby="nonexistent_col", n_obs_per_chunk=2, zarr_shard_size=10, n_obs_per_dataset=200
        )


# =============================================================================
# Loader integration
# =============================================================================


def test_loader_with_grouped_collection(
    adata_with_h5_path_different_var_space: tuple[ad.AnnData, Path], tmp_path: Path
):
    paths = sorted(p for p in adata_with_h5_path_different_var_space[1].iterdir() if p.suffix == ".h5ad")[:3]
    grouped = GroupedCollection(tmp_path / "grouped_loader.zarr").add_adatas(
        paths,
        groupby=["label", "store_id"],
        n_obs_per_chunk=2,
        zarr_shard_size=10,
        n_obs_per_dataset=200,
        random_seed=0,
    )
    loader = Loader(
        chunk_size=10,
        preload_nchunks=4,
        batch_size=20,
        shuffle=False,
        return_index=True,
        preload_to_gpu=False,
        to_torch=False,
    ).use_collection(grouped)

    all_indices = []
    for batch in loader:
        assert batch["X"].shape[0] > 0
        all_indices.append(batch["index"])
    stacked = np.concatenate(all_indices)
    assert len(stacked) == loader.n_obs
    assert np.array_equal(np.sort(stacked), np.arange(loader.n_obs))


# =============================================================================
# CategoricalSampler.from_collection
# =============================================================================


def test_categorical_sampler_from_collection(
    adata_with_h5_path_different_var_space: tuple[ad.AnnData, Path], tmp_path: Path
):
    paths = sorted(p for p in adata_with_h5_path_different_var_space[1].iterdir() if p.suffix == ".h5ad")[:3]
    grouped = GroupedCollection(tmp_path / "grouped_catsampler.zarr").add_adatas(
        paths,
        groupby="label",
        n_obs_per_chunk=2,
        zarr_shard_size=10,
        n_obs_per_dataset=200,
        shuffle=False,
    )
    group_index = grouped.group_index
    min_cat_size = int(group_index["count"].min())
    chunk_size = min(min_cat_size, 2)
    batch_size = chunk_size
    sampler = CategoricalSampler.from_collection(
        grouped,
        chunk_size=chunk_size,
        preload_nchunks=2,
        batch_size=batch_size,
        num_samples=20 * batch_size,
        rng=np.random.default_rng(0),
    )
    assert sampler.n_categories == len(group_index)
    assert sampler.n_iters(0) == 20

    total_batches = 0
    for lr in sampler.sample(int(group_index["stop"].iloc[-1])):
        total_batches += len(lr["splits"])
    assert total_batches == 20


# =============================================================================
# End-to-end: batch purity
# =============================================================================


def _build_grouped_collection(paths, tmp_path, name, groupby="label"):
    return GroupedCollection(tmp_path / f"{name}.zarr").add_adatas(
        paths,
        groupby=groupby,
        n_obs_per_chunk=2,
        zarr_shard_size=10,
        n_obs_per_dataset=2_000_000,
        shuffle=False,
    )


def test_e2e_each_batch_is_pure_category(
    adata_with_h5_path_different_var_space: tuple[ad.AnnData, Path], tmp_path: Path
):
    """Every batch yielded by Loader+CategoricalSampler contains obs from exactly one label group."""
    paths = sorted(p for p in adata_with_h5_path_different_var_space[1].iterdir() if p.suffix == ".h5ad")[:3]
    grouped = _build_grouped_collection(paths, tmp_path, "e2e_pure")
    group_index = grouped.group_index

    min_cat_size = int(group_index["count"].min())
    chunk_size = min(min_cat_size, 2)
    batch_size = chunk_size
    n_batches = 30

    sampler = CategoricalSampler.from_collection(
        grouped,
        chunk_size=chunk_size,
        preload_nchunks=2,
        batch_size=batch_size,
        num_samples=n_batches * batch_size,
        rng=np.random.default_rng(42),
    )
    loader = Loader(
        batch_sampler=sampler,
        return_index=False,
        preload_to_gpu=False,
        to_torch=False,
    ).use_collection(grouped)

    batch_count = 0
    for batch in loader:
        obs = batch["obs"]
        assert obs is not None
        labels_in_batch = obs["label"].unique()
        assert len(labels_in_batch) == 1, f"Batch has multiple labels: {labels_in_batch.tolist()}"
        batch_count += 1
    assert batch_count == n_batches


def test_e2e_batch_larger_than_category(
    adata_with_h5_path_different_var_space: tuple[ad.AnnData, Path], tmp_path: Path
):
    """batch_size > smallest category works via with-replacement sampling."""
    paths = sorted(p for p in adata_with_h5_path_different_var_space[1].iterdir() if p.suffix == ".h5ad")[:3]
    grouped = _build_grouped_collection(paths, tmp_path, "e2e_big_batch")
    group_index = grouped.group_index

    min_cat_size = int(group_index["count"].min())
    chunk_size = min(min_cat_size, 2)
    batch_size = min_cat_size * 2
    # preload_nchunks must satisfy: chunk_size * preload_nchunks >= batch_size
    # and (chunk_size * preload_nchunks) % batch_size == 0
    preload_nchunks = batch_size // chunk_size
    n_batches = 10

    sampler = CategoricalSampler.from_collection(
        grouped,
        chunk_size=chunk_size,
        preload_nchunks=preload_nchunks,
        batch_size=batch_size,
        num_samples=n_batches * batch_size,
        rng=np.random.default_rng(7),
    )
    loader = Loader(
        batch_sampler=sampler,
        return_index=False,
        preload_to_gpu=False,
        to_torch=False,
    ).use_collection(grouped)

    batch_count = 0
    for batch in loader:
        obs = batch["obs"]
        assert obs is not None
        labels_in_batch = obs["label"].unique()
        assert len(labels_in_batch) == 1, (
            f"Batch has mixed labels: {labels_in_batch.tolist()}"
        )
        assert batch["X"].shape[0] == batch_size
        batch_count += 1
    assert batch_count == n_batches


def test_e2e_weighted_categories_bias(
    adata_with_h5_path_different_var_space: tuple[ad.AnnData, Path], tmp_path: Path
):
    """Heavily weighted category appears in most batches."""
    paths = sorted(p for p in adata_with_h5_path_different_var_space[1].iterdir() if p.suffix == ".h5ad")[:3]
    grouped = _build_grouped_collection(paths, tmp_path, "e2e_weighted")
    group_index = grouped.group_index
    n_categories = len(group_index)

    min_cat_size = int(group_index["count"].min())
    chunk_size = min(min_cat_size, 2)
    batch_size = chunk_size
    n_batches = 100

    weights = np.zeros(n_categories)
    weights[0] = 1.0

    sampler = CategoricalSampler.from_collection(
        grouped,
        chunk_size=chunk_size,
        preload_nchunks=2,
        batch_size=batch_size,
        num_samples=n_batches * batch_size,
        weights=weights,
        rng=np.random.default_rng(0),
    )
    loader = Loader(
        batch_sampler=sampler,
        return_index=False,
        preload_to_gpu=False,
        to_torch=False,
    ).use_collection(grouped)

    dominant_label = str(group_index.iloc[0]["label"])
    dominant_count = 0
    total = 0
    for batch in loader:
        label = batch["obs"]["label"].iloc[0]
        if str(label) == dominant_label:
            dominant_count += 1
        total += 1

    assert dominant_count == total, (
        f"Expected all {total} batches to be label={dominant_label!r}, got {dominant_count}"
    )


def test_multistore_grouped_collection_uses_filename_stem(
    adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path],
):
    paths = sorted(p for p in adata_with_zarr_path_same_var_space[1].iterdir() if p.suffix == ".zarr")
    collection = GroupedCollection(paths)

    group_index = collection.group_index
    assert list(group_index["category"]) == [p.stem for p in paths]
    assert int(group_index["start"].iloc[0]) == 0
    assert np.array_equal(
        group_index["count"].to_numpy(),
        group_index["stop"].to_numpy() - group_index["start"].to_numpy(),
    )
    assert int(group_index["count"].sum()) == adata_with_zarr_path_same_var_space[0].n_obs


def test_multistore_grouped_collection_prefers_metadata_category(
    adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path], tmp_path: Path
):
    source_paths = sorted(p for p in adata_with_zarr_path_same_var_space[1].iterdir() if p.suffix == ".zarr")
    copied_paths = []
    categories = []
    for i, source_path in enumerate(source_paths):
        target = tmp_path / f"store_{i}.zarr"
        copied_paths.append(target)
        category = f"cell_line_{i}"
        categories.append(category)
        group = zarr.open_group(target, mode="w", zarr_format=3)
        write_sharded(group, ad.read_zarr(source_path), n_obs_per_chunk=10, shard_size=20)
        group.attrs["category"] = category

    collection = GroupedCollection(copied_paths)
    assert list(collection.group_index["category"]) == categories


def test_virtual_grouped_collection_add_adatas_matches_multistore_boundaries(
    adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path],
):
    paths = sorted(p for p in adata_with_zarr_path_same_var_space[1].iterdir() if p.suffix == ".zarr")

    virtual = GroupedCollection().add_adatas(paths)
    multi_store = GroupedCollection(paths)

    pd.testing.assert_frame_equal(
        virtual.group_index[["category", "count", "start", "stop"]].reset_index(drop=True),
        multi_store.group_index[["category", "count", "start", "stop"]].reset_index(drop=True),
    )


def test_virtual_grouped_collection_uses_explicit_labels_for_in_memory_inputs():
    adatas = [
        ad.AnnData(X=np.ones((2, 3), dtype=np.float32)),
        ad.AnnData(X=np.ones((4, 3), dtype=np.float32)),
    ]

    collection = GroupedCollection().add_adatas(adatas, category_labels=["a", "b"])
    group_index = collection.group_index

    assert list(group_index["category"]) == ["a", "b"]
    assert list(group_index["count"]) == [2, 4]


def test_virtual_grouped_collection_requires_labels_for_in_memory_inputs_without_metadata():
    adatas = [
        ad.AnnData(X=np.ones((2, 3), dtype=np.float32)),
        ad.AnnData(X=np.ones((4, 3), dtype=np.float32)),
    ]

    with pytest.raises(ValueError, match="category_labels"):
        GroupedCollection().add_adatas(adatas)


def test_loader_with_virtual_grouped_collection(
    adata_with_zarr_path_same_var_space: tuple[ad.AnnData, Path],
):
    paths = sorted(p for p in adata_with_zarr_path_same_var_space[1].iterdir() if p.suffix == ".zarr")
    collection = GroupedCollection().add_adatas(paths)

    loader = Loader(
        chunk_size=10,
        preload_nchunks=2,
        batch_size=20,
        shuffle=False,
        return_index=True,
        preload_to_gpu=False,
        to_torch=False,
    ).use_collection(collection)

    all_indices = []
    for batch in loader:
        all_indices.append(batch["index"])
    stacked = np.concatenate(all_indices)
    assert len(stacked) == loader.n_obs
    assert np.array_equal(np.sort(stacked), np.arange(loader.n_obs))

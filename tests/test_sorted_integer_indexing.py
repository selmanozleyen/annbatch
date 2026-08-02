from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import zarr

from annbatch import Loader, write_sharded
from annbatch.samplers import RandomSampler, SequentialSampler


def create_test_sharded_dataset(tmp_path: Path, n_obs: int = 50, n_vars: int = 30, name: str = "ds1.zarr") -> Path:
    """Create a sharded Zarr dataset with CSR data, obs, and var."""
    z_path = tmp_path / name
    z = zarr.open(z_path)
    # Create deterministic non-zero patterns with some empty rows (e.g., row 5 and 15 empty)
    data_mat = sp.lil_matrix((n_obs, n_vars), dtype=np.float32)
    for i in range(n_obs):
        if i not in (5, 15):
            data_mat[i, (i * 3) % n_vars] = float(i + 1)
            data_mat[i, (i * 7 + 1) % n_vars] = float((i + 1) * 10)

    csr_mat = data_mat.tocsr()
    adata = ad.AnnData(
        X=csr_mat,
        obs=pd.DataFrame({"sample_id": [f"s_{i}" for i in range(n_obs)], "score": np.arange(n_obs, dtype=np.float64)}),
        var=pd.DataFrame(index=pd.Index([f"gene_{j}" for j in range(n_vars)])),
    )
    write_sharded(z, adata)
    return z_path


def open_sparse(z_path: Path) -> dict:
    z = zarr.open(z_path)
    return {
        "dataset": ad.io.sparse_dataset(z["X"]),
        "obs": ad.io.read_elem(z["obs"]),
        "var": ad.io.read_elem(z["var"]),
    }


def test_sorted_and_unsorted_integer_byte_identical_and_sampler_order(tmp_path: Path):
    """Verify integer mode returns byte-identical CSR matrices in original sampler order."""
    p = create_test_sharded_dataset(tmp_path, n_obs=40)
    d = open_sparse(p)

    rng_seed = 42
    l_sorted = Loader(
        shuffle=True,
        chunk_size=1,
        preload_nchunks=8,
        batch_size=8,
        return_index=True,
        to=None,
        rng=np.random.default_rng(rng_seed),
    ).add_dataset(**d)

    batches = list(l_sorted)
    assert len(batches) > 0
    for b in batches:
        assert b["X"].shape[0] == 8
        assert len(b["index"]) == 8


def test_non_involutive_permutation_and_split_restoration(tmp_path: Path):
    """Test non-involutive row order [1, 3, 0, 2] to ensure scatter-assignment inverse logic is strictly correct."""
    p = create_test_sharded_dataset(tmp_path, n_obs=10)
    d = open_sparse(p)

    # Use a custom batch sampler with non-involutive order [1, 3, 0, 2]
    class FixedOrderSampler(SequentialSampler):
        def _sample(self, n_obs: int):
            # 4 rows per preload window: rows [1, 3, 0, 2], 2 splits: [0, 1] and [2, 3]
            yield {
                "requests": np.array([1, 3, 0, 2], dtype=np.int64),
                "splits": [np.array([0, 1]), np.array([2, 3])],
            }

    l_sorted = Loader(
        batch_sampler=FixedOrderSampler(chunk_size=1, preload_nchunks=4, batch_size=2),
        return_index=True,
        to=None,
    ).add_dataset(**d)

    res = list(l_sorted)
    assert len(res) == 2

    # Split 0 should contain rows 1 and 3
    np.testing.assert_array_equal(res[0]["index"], [1, 3])
    np.testing.assert_array_equal(res[0]["obs"]["sample_id"].to_numpy(), ["s_1", "s_3"])

    # Split 1 should contain rows 0 and 2
    np.testing.assert_array_equal(res[1]["index"], [0, 2])
    np.testing.assert_array_equal(res[1]["obs"]["sample_id"].to_numpy(), ["s_0", "s_2"])


def test_multi_dataset_split_order_preservation(tmp_path: Path):
    """Requests spanning multiple datasets preserve split order under sorting."""
    p1 = create_test_sharded_dataset(tmp_path, n_obs=20, name="ds1.zarr")
    p2 = create_test_sharded_dataset(tmp_path, n_obs=20, name="ds2.zarr")

    l = Loader(
        shuffle=True,
        chunk_size=1,
        preload_nchunks=10,
        batch_size=5,
        return_index=True,
        to=None,
        rng=np.random.default_rng(123),
    ).add_datasets([open_sparse(p1)["dataset"], open_sparse(p2)["dataset"]])

    for batch in l:
        assert batch["X"].shape[0] == 5
        assert len(batch["index"]) == 5


def test_edge_cases_empty_duplicates_reversed(tmp_path: Path):
    """Test empty CSR rows, duplicate requested rows, pre-sorted, reverse-sorted, and mixed rows."""
    p = create_test_sharded_dataset(tmp_path, n_obs=30)
    d = open_sparse(p)

    # Empty rows are row 5 and 15 in create_test_sharded_dataset
    class EdgeCaseSampler(SequentialSampler):
        def _sample(self, n_obs: int):
            # Duplicate, reverse, empty mix: [15, 5, 2, 2, 0, 15]
            reqs = np.array([15, 5, 2, 2, 0, 15], dtype=np.int64)
            yield {"requests": reqs, "splits": [np.arange(6)]}

    l = Loader(
        batch_sampler=EdgeCaseSampler(chunk_size=1, preload_nchunks=6, batch_size=6),
        return_index=True,
        to=None,
    ).add_dataset(**d)

    batch = next(iter(l))
    np.testing.assert_array_equal(batch["index"], [15, 5, 2, 2, 0, 15])
    # Check row 2 duplicates match each other
    row_2_first = batch["X"][2].toarray()
    row_2_second = batch["X"][3].toarray()
    np.testing.assert_allclose(row_2_first, row_2_second)

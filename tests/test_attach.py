"""Tests for :meth:`annbatch.io.DatasetCollection.attach` (symlink existing zarr stores)."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import zarr

from annbatch import write_sharded
from annbatch.io import DATASET_PREFIX, V1_ENCODING, DatasetCollection


def _write_adata_zarr(path: Path, n_obs: int, var_names, seed: int) -> ad.AnnData:
    """Write a tiny sharded AnnData zarr and return the in-memory AnnData."""
    rng = np.random.default_rng(seed)
    X = sp.random(n_obs, len(var_names), density=0.25, format="csr", dtype="float32", random_state=seed)
    obs = pd.DataFrame(
        {"cell_type": pd.Categorical(rng.choice(["A", "B", "C"], n_obs))},
        index=[f"s{seed}_c{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=pd.Index(list(var_names), name="gene"))
    adata = ad.AnnData(X=X, obs=obs, var=var)
    write_sharded(zarr.open_group(path, mode="w"), adata, n_obs_per_chunk=8, shard_size="1MB")
    return adata


def test_attach_links_and_streams(tmp_path):
    vs = [f"g{i}" for i in range(24)]
    _write_adata_zarr(tmp_path / "p0.zarr", 30, vs, 0)
    _write_adata_zarr(tmp_path / "p1.zarr", 40, vs, 1)
    coll_path = tmp_path / "coll.zarr"

    coll = DatasetCollection(coll_path)
    assert coll.is_empty
    returned = coll.attach([tmp_path / "p0.zarr", tmp_path / "p1.zarr"])
    assert returned is coll  # chainable

    # linked as symlinks (no data copy) + marked preshuffled
    assert (coll_path / f"{DATASET_PREFIX}_0").is_symlink()
    assert (coll_path / f"{DATASET_PREFIX}_1").is_symlink()
    assert (coll_path / f"{DATASET_PREFIX}_0").resolve() == (tmp_path / "p0.zarr").resolve()
    assert V1_ENCODING.items() <= zarr.open_group(coll_path, mode="r").attrs.items()

    # reads like a normal collection (fresh handle == on-disk truth)
    reopened = DatasetCollection(coll_path, mode="r")
    assert not reopened.is_empty
    assert len(list(reopened)) == 2
    assert len(reopened.obs(columns=["cell_type"])) == 70


def test_attach_rejects_var_mismatch(tmp_path):
    _write_adata_zarr(tmp_path / "p0.zarr", 12, [f"g{i}" for i in range(24)], 0)
    _write_adata_zarr(tmp_path / "p1.zarr", 12, [f"h{i}" for i in range(24)], 1)  # disjoint var
    coll_path = tmp_path / "coll.zarr"
    coll = DatasetCollection(coll_path)

    with pytest.raises(ValueError, match="var"):
        coll.attach([tmp_path / "p0.zarr", tmp_path / "p1.zarr"])

    # integrity check runs before any linking -> nothing created, still empty
    assert not (coll_path / f"{DATASET_PREFIX}_0").exists()
    assert DatasetCollection(coll_path, mode="r").is_empty


def test_attach_appends_to_existing(tmp_path):
    vs = [f"g{i}" for i in range(16)]
    for i in range(3):
        _write_adata_zarr(tmp_path / f"p{i}.zarr", 10 + i, vs, i)
    coll_path = tmp_path / "coll.zarr"

    coll = DatasetCollection(coll_path)
    coll.attach([tmp_path / "p0.zarr"])
    coll.attach([tmp_path / "p1.zarr", tmp_path / "p2.zarr"])  # numbered after existing

    assert coll._dataset_keys == [f"{DATASET_PREFIX}_{i}" for i in range(3)]
    assert len(DatasetCollection(coll_path, mode="r").obs(columns=["cell_type"])) == (10 + 11 + 12)


def test_attach_missing_source_raises(tmp_path):
    coll = DatasetCollection(tmp_path / "coll.zarr")
    with pytest.raises(FileNotFoundError):
        coll.attach([tmp_path / "nope.zarr"])
    assert DatasetCollection(tmp_path / "coll.zarr", mode="r").is_empty

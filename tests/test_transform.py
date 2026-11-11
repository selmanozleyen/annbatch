from __future__ import annotations

import pickle
from contextlib import nullcontext
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import zarr

from annbatch import Loader, RemapColumns

if TYPE_CHECKING:
    from pathlib import Path

# Overlapping, unsorted, and in a different order per dataset: the union is not any one dataset's order.
VARS = [["g3", "g1", "g0"], ["g0", "g4", "g2", "g3"], ["g5", "g1"]]
UNION = pd.Index(["g0", "g1", "g2", "g3", "g4", "g5"])


def _adata(var_names: list[str], n_obs: int, dtype: str, seed: int) -> ad.AnnData:
    x = sp.random(n_obs, len(var_names), density=0.4, format="csr", dtype=dtype, rng=np.random.default_rng(seed))
    return ad.AnnData(X=x, var=pd.DataFrame(index=var_names))


def _datasets(tmp_path: Path, *, backed: bool, dtypes: list[str]) -> list[ad.AnnData]:
    adatas = [_adata(v, n_obs=10 + 5 * i, dtype=d, seed=i) for i, (v, d) in enumerate(zip(VARS, dtypes, strict=True))]
    if not backed:
        return adatas
    out = []
    for i, a in enumerate(adatas):
        g = zarr.open_group(tmp_path / f"d{i}.zarr", mode="w")
        ad.io.write_elem(g, "X", a.X)
        out.append(ad.AnnData(X=ad.io.sparse_dataset(g["X"]), var=a.var))
    return out


def _loader(**kwargs) -> Loader:
    return Loader(chunk_size=5, preload_nchunks=2, batch_size=5, preload_to_gpu=False, **kwargs)


def _read_all(loader: Loader) -> np.ndarray:
    xs = [batch["X"] for batch in loader]
    return sp.vstack(xs).toarray() if sp.issparse(xs[0]) else np.concatenate(xs)


@pytest.mark.parametrize("backed", [True, False], ids=["backed", "in-memory"])
@pytest.mark.parametrize(
    "dtypes", [["float32"] * 3, ["float32", "float64", "float32"]], ids=["same-dtype", "mixed-dtype"]
)
def test_remap_onto_union_matches_outer_concat(tmp_path: Path, backed: bool, dtypes: list[str]):
    adatas = _datasets(tmp_path, backed=backed, dtypes=dtypes)
    loader = _loader(var=pd.DataFrame(index=UNION))
    mixed = len(set(dtypes)) > 1
    with pytest.warns(UserWarning, match="Heterogeneous dtypes") if mixed else nullcontext():
        loader.add_adatas(adatas, transforms=[RemapColumns(a.var_names, UNION) for a in adatas])

    expected = ad.concat([a.to_memory() if backed else a for a in adatas], join="outer")[:, UNION]
    assert loader.n_var == len(UNION)
    np.testing.assert_allclose(_read_all(loader), expected.X.toarray())
    assert next(iter(loader))["var"].index.equals(UNION)


def test_remap_keeps_unmeasured_implicit_and_records_col_map():
    a = _adata(VARS[1], 12, "float32", 0)
    remap = RemapColumns(a.var_names, UNION)
    assert list(UNION[remap.col_map]) == VARS[1]
    out = remap(a.X)
    assert out.shape == (12, len(UNION)) and out.nnz == a.X.nnz  # no stored zeros for g1, g5
    np.testing.assert_array_equal(out.toarray()[:, remap.col_map], a.X.toarray())


def test_remap_is_picklable():
    remap = pickle.loads(pickle.dumps(RemapColumns(VARS[0], UNION)))
    np.testing.assert_array_equal(remap.col_map, UNION.get_indexer(VARS[0]))


@pytest.mark.parametrize("dense", [True, False], ids=["dense", "sparse"])
def test_width_keeping_transform_without_var(dense: bool):
    x = np.random.default_rng(0).random((20, 3), dtype="float32")
    loader = _loader()
    loader.add_dataset(x if dense else sp.csr_matrix(x), transform=_double)
    loader.add_dataset(x if dense else sp.csr_matrix(x))
    np.testing.assert_allclose(_read_all(loader), np.concatenate([2 * x, x]))


def _double(x):  # module level, so picklable
    return x * 2


@pytest.mark.parametrize("backed", [True, False], ids=["backed", "in-memory"])
def test_declared_var_remaps_by_name_by_default(tmp_path: Path, backed: bool):
    adatas = _datasets(tmp_path, backed=backed, dtypes=["float32"] * 3)
    loader = _loader(var=pd.DataFrame(index=UNION))
    loader.add_adatas(adatas)  # no transforms: each dataset is placed by its var_names
    expected = ad.concat([a.to_memory() if backed else a for a in adatas], join="outer")[:, UNION]
    np.testing.assert_allclose(_read_all(loader), expected.X.toarray())


def test_same_width_other_order_is_placed_by_name():
    a = _adata(["g2", "g0", "g1"], 10, "float32", 0)
    loader = _loader(var=pd.DataFrame(index=["g0", "g1", "g2"]))
    loader.add_adata(a)
    np.testing.assert_allclose(_read_all(loader), a[:, ["g0", "g1", "g2"]].X.toarray())


def test_feature_missing_from_declared_var_raises():
    loader = _loader(var=pd.DataFrame(index=UNION[:3]))
    with pytest.raises(ValueError, match="not in target"):
        loader.add_adata(_adata(VARS[1], 10, "float32", 0))


def test_dense_with_other_var_names_needs_transform():
    loader = _loader(var=pd.DataFrame(index=["g0", "g1"]))
    dense = ad.AnnData(X=np.zeros((10, 2), dtype="float32"), var=pd.DataFrame(index=["g1", "g0"]))
    with pytest.raises(ValueError, match="it is dense"):
        loader.add_adata(dense)


def test_declared_var_needs_transform_for_other_width_without_var():
    loader = _loader(var=pd.DataFrame(index=UNION))
    with pytest.raises(ValueError, match="give it a transform"):
        loader.add_dataset(_adata(VARS[0], 10, "float32", 0).X)


@pytest.mark.parametrize("method", ["add_adatas", "add_datasets"])
def test_wrong_transforms_length_adds_nothing(method: str):
    adatas = [_adata(VARS[0], 10, "float32", i) for i in range(3)]
    loader = _loader()
    items = adatas if method == "add_adatas" else [a.X for a in adatas]
    with pytest.raises(ValueError, match="transforms has 1 entries for 3 datasets"):
        getattr(loader, method)(items, transforms=[None])
    with pytest.raises(ValueError, match="No datasets added yet"):
        loader.n_var  # noqa: B018


def test_declared_var_accepts_matching_width_without_transform():
    loader = _loader(var=pd.DataFrame(index=UNION))
    loader.add_adata(_adata(list(UNION), 10, "float32", 0))
    assert _read_all(loader).shape == (10, len(UNION))


def test_transform_output_is_checked():
    loader = _loader(var=pd.DataFrame(index=UNION))
    loader.add_adata(_adata(VARS[0], 10, "float32", 0), transform=RemapColumns(VARS[0], UNION[:4]))
    with pytest.raises(ValueError, match=r"must return CSR of shape \(\d+, 6\)"):
        _read_all(loader)


def test_transform_must_be_callable():
    with pytest.raises(TypeError, match="transform must be callable"):
        _loader().add_adata(_adata(VARS[0], 10, "float32", 0), transform="log1p")  # type: ignore[arg-type]


def test_remap_rejects_unknown_and_duplicate_features():
    with pytest.raises(ValueError, match="not in target"):
        RemapColumns(["g0", "gX"], UNION)
    with pytest.raises(ValueError, match="source feature names must be unique"):
        RemapColumns(["g0", "g0"], UNION)


def test_without_var_different_widths_still_raise():
    loader = _loader()
    loader.add_adata(_adata(VARS[0], 10, "float32", 0))
    with pytest.raises(ValueError, match="same shape along the var axis"):
        loader.add_adata(_adata(VARS[1], 10, "float32", 1))

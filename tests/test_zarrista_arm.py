"""The zarrista arm returns what the zarr-python arm returns, and is actually zarrista.

Equality against the arm already trusted, not against a hand-computed expectation -- which
would be a second thing that can be wrong. And a call counter beside it, because an arm that
silently fell back to zarr-python would pass every equality test ever written and be worthless.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

zarrista = pytest.importorskip("zarrista")

import anndata as ad  # noqa: E402
import zarr  # noqa: E402

from annbatch import Loader  # noqa: E402
from annbatch._zarrista import ZarristaCSRElems, coalesce  # noqa: E402
from annbatch.samplers import RandomSampler  # noqa: E402

N_COLS = 64


@pytest.fixture
def plates(tmp_path):
    """Three CSR plates, one of them with empty rows -- including the first and the last."""
    paths = []
    for plate in range(3):
        x = sp.random(400, N_COLS, density=0.08, format="csr", random_state=plate,
                      dtype=np.float32)
        if plate == 1:
            # Empty rows are the case the offset bookkeeping is most likely to get wrong, and
            # the first and last rows are where an off-by-one shows up.
            lil = x.tolil()
            for r in (0, 7, 14, 399):
                lil.rows[r], lil.data[r] = [], []
            x = lil.tocsr()
        path = tmp_path / f"plate{plate}.zarr"
        ad.AnnData(X=x).write_zarr(path)
        paths.append(path)
    return paths


def _read(paths, *, use_zarrista, chunk_size, seed=17, rows=384):
    datasets = [ad.io.sparse_dataset(zarr.open_group(p, mode="r")["X"]) for p in paths]
    # the preload must hold a batch: at chunk_size=1 eight chunks is eight rows
    sampler = RandomSampler(chunk_size=chunk_size, preload_nchunks=max(8, 64 // chunk_size),
                            batch_size=32,
                            replacement=False, rng=np.random.default_rng(seed))
    loader = Loader(batch_sampler=sampler, preload_to_gpu=False, to=None,
                    use_zarrista=use_zarrista)
    loader.add_datasets(datasets)
    out, n = [], 0
    for batch in loader:
        x = batch["X"] if isinstance(batch, dict) else batch
        arr = x.toarray() if sp.issparse(x) else np.asarray(x)
        out.append(arr)
        n += arr.shape[0]
        if n >= rows:
            break
    return np.concatenate(out, axis=0), loader


@pytest.mark.parametrize("chunk_size", [1, 4, 32])
def test_arms_agree(plates, chunk_size):
    """Both arms return identical bytes, at every draw granularity."""
    expected, _ = _read(plates, use_zarrista=False, chunk_size=chunk_size)
    actual, _ = _read(plates, use_zarrista=True, chunk_size=chunk_size)
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)


def test_zarrista_actually_ran(plates, monkeypatch):
    """The arm is the arm.

    Patched at the zarrista boundary, not at `Loader._fetch_data_zarrista`:
    `singledispatchmethod` captured the function at decoration time, so patching the method
    counts nothing while zarrista is demonstrably being called.
    """
    calls = {"n": 0}
    original = zarrista.Array.retrieve_array_subset

    def counted(self, selection):
        calls["n"] += 1
        return original(self, selection)

    monkeypatch.setattr(zarrista.Array, "retrieve_array_subset", counted)
    _, loader = _read(plates, use_zarrista=True, chunk_size=4)
    assert calls["n"] > 0, "use_zarrista=True but zarrista was never called"
    assert isinstance(loader._get_elem_from_cache(0), ZarristaCSRElems)


def test_flag_refuses_what_it_cannot_serve(tmp_path):
    """A knob that was set must be a knob that arrived: dense has no zarrista path yet."""
    z = zarr.create_array(store=tmp_path / "dense.zarr", name="X", shape=(64, N_COLS),
                          chunks=(16, N_COLS), dtype="float32", overwrite=True)
    z[:] = np.zeros((64, N_COLS), dtype=np.float32)
    loader = Loader(chunk_size=4, preload_nchunks=4, batch_size=8, shuffle=True,
                    preload_to_gpu=False, to=None, use_zarrista=True)
    with pytest.raises(TypeError, match="only serves backed CSR"):
        loader.add_datasets([z])


def test_coalesce_is_order_independent():
    """Runs arrive shuffled -- `_group_rows` orders by dataset, not by row.

    A merged range that assumed ascending input produced a negative index into its own
    result and silently read nothing, which numpy reported as a broadcast error three frames
    away. Each run therefore carries the offset it must land at.
    """
    unit = 100
    runs = [(0, slice(250, 260)), (10, slice(10, 20)), (20, slice(255, 265)), (30, slice(0, 5))]
    groups = coalesce(runs, unit)
    # every non-empty run survives exactly once, with its offset intact
    seen = sorted((off, s.start, s.stop) for _, members in groups for off, s in members)
    assert seen == [(0, 250, 260), (10, 10, 20), (20, 255, 265), (30, 0, 5)]
    # and every member lies inside the range it was merged into
    for merged, members in groups:
        for _, limit in members:
            assert merged.start <= limit.start and limit.stop <= merged.stop


def test_coalesce_merges_within_a_unit_only():
    """Merging must never pull in a decode unit that was not already needed."""
    unit = 100
    # two runs in unit 0, one far away in unit 5
    groups = coalesce([(0, slice(0, 10)), (10, slice(50, 60)), (20, slice(500, 510))], unit)
    assert len(groups) == 2
    assert groups[0][0] == slice(0, 60)
    assert groups[1][0] == slice(500, 510)


def test_empty_runs_do_not_shift_offsets():
    """An empty row reads nothing but must not move what follows it."""
    groups = coalesce([(0, slice(5, 5)), (0, slice(10, 20)), (10, slice(30, 30))], 100)
    members = [(off, s.start, s.stop) for _, mem in groups for off, s in mem]
    assert members == [(0, 10, 20)]

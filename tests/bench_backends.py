"""Head-to-head benchmark: zarrs (Rust+mmap) vs C ext (mmap) vs zarr-python (async).

Uses annbatch's Loader end-to-end with the sync_backend switch.

Run with:
    python tests/bench_backends.py
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import zarr

from annbatch import ChunkSampler, Loader, write_sharded
from annbatch._direct_read import _HAS_C_EXT

N_OBS = 10_000
N_VARS = 200
REPEATS = 3


def _create_store(tmp: Path) -> Path:
    store_path = tmp / "bench.zarr"
    rng = np.random.default_rng(0)
    adata = ad.AnnData(
        X=rng.standard_normal((N_OBS, N_VARS)).astype("f4"),
        obs=pd.DataFrame(
            {"label": rng.integers(0, 5, size=N_OBS)},
            index=[str(i) for i in range(N_OBS)],
        ),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(N_VARS)]),
        layers={
            "sparse": sp.random(
                N_OBS,
                N_VARS,
                density=0.1,
                format="csr",
                dtype="f4",
                random_state=np.random.default_rng(0),
            ),
        },
    )
    g = zarr.open_group(store_path, mode="w", zarr_format=3)
    write_sharded(
        g,
        adata,
        dense_chunk_size=64,
        dense_shard_size=128,
        sparse_chunk_size=10,
        sparse_shard_size=20,
    )
    return store_path


def _open_dense(path: Path) -> ad.AnnData:
    g = zarr.open(path)
    return ad.AnnData(
        X=g["X"],
        obs=ad.io.read_elem(g["obs"]),
        var=ad.io.read_elem(g["var"]),
    )


def _open_sparse(path: Path) -> ad.AnnData:
    g = zarr.open(path)
    return ad.AnnData(
        X=ad.io.sparse_dataset(g["layers"]["sparse"]),
        obs=ad.io.read_elem(g["obs"]),
        var=ad.io.read_elem(g["var"]),
    )


def bench_loader(store_path, chunk_size, preload_nchunks, batch_size, sparse,
                  sync_backend, fuse_ranges=True):
    if sync_backend == "zarrs":
        import zarr_direct.reader as _zdr
        _zdr._reader_cache.clear()
        _zdr._FUSE_DEFAULT = fuse_ranges

    adata = _open_sparse(store_path) if sparse else _open_dense(store_path)
    sampler = ChunkSampler(
        chunk_size=chunk_size,
        preload_nchunks=preload_nchunks,
        batch_size=batch_size,
        shuffle=False,
    )
    loader = Loader(
        batch_sampler=sampler,
        return_index=True,
        preload_to_gpu=False,
        to_torch=False,
        sync_backend=sync_backend,
    )
    loader.add_anndata(adata)

    n_batches = 0
    total_obs = 0
    t0 = time.perf_counter()
    for batch in loader:
        n_batches += 1
        total_obs += batch["X"].shape[0]
    elapsed = time.perf_counter() - t0
    return {"elapsed": elapsed, "n_batches": n_batches, "total_obs": total_obs}


def run_suite(store_path, label, sparse, configs):
    backends = [
        ("zarrs (fuse=on)", "zarrs", True),
        ("zarrs (fuse=off)", "zarrs", False),
        ("C ext (mmap)", "c", True),
        ("zarr-python async", None, True),
    ]

    print(f"\n{'=' * 110}")
    print(f"  {label}  --  {N_OBS} obs x {N_VARS} vars, repeats={REPEATS}, C ext={_HAS_C_EXT}")
    print(f"{'=' * 110}")

    for cfg in configs:
        cs, pn, bs = cfg["chunk_size"], cfg["preload_nchunks"], cfg["batch_size"]
        print(f"\n  chunk_size={cs:>3}  preload={pn:>3}  batch={bs:>3}")
        print(f"  {'-' * 100}")
        print(
            f"  {'Backend':<25}  {'best (s)':>10}  {'mean (s)':>10}  {'obs/s':>12}  {'vs async':>10}  {'vs C ext':>10}"
        )
        print(f"  {'-' * 100}")

        results = {}
        for name, sb, fuse in backends:
            if sb == "c" and not _HAS_C_EXT:
                results[name] = None
                continue
            times = []
            res = None
            for _ in range(REPEATS):
                res = bench_loader(store_path, sparse=sparse, sync_backend=sb,
                                    fuse_ranges=fuse, **cfg)
                times.append(res["elapsed"])
            best = min(times)
            mean = sum(times) / len(times)
            obs_per_sec = res["total_obs"] / best
            results[name] = (best, mean, obs_per_sec, res["total_obs"])

        async_best = results.get("zarr-python async")
        c_best = results.get("C ext (mmap)")

        for name, _, _ in backends:
            vals = results.get(name)
            if vals is None:
                print(f"  {name:<25}  {'N/A':>10}  {'N/A':>10}  {'N/A':>12}  {'N/A':>10}  {'N/A':>10}")
                continue
            best, mean, obs_per_sec, _ = vals
            vs_async = f"{async_best[0] / best:.2f}x" if async_best else "N/A"
            vs_c = f"{c_best[0] / best:.2f}x" if c_best else "N/A"
            print(f"  {name:<25}  {best:>10.4f}  {mean:>10.4f}  {obs_per_sec:>12.0f}  {vs_async:>10}  {vs_c:>10}")

    print()


def main():
    configs = [
        {"chunk_size": 2, "preload_nchunks": 512, "batch_size": 8},
        {"chunk_size": 2, "preload_nchunks": 512, "batch_size": 16},
        {"chunk_size": 2, "preload_nchunks": 512, "batch_size": 32},
    ]

    with tempfile.TemporaryDirectory() as tmp:
        store_path = _create_store(Path(tmp))
        run_suite(store_path, "Dense (f32)", sparse=False, configs=configs)
        run_suite(store_path, "Sparse CSR (f32, density=0.1)", sparse=True, configs=configs)

    print("Done.")


if __name__ == "__main__":
    main()

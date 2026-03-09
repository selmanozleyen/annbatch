"""Benchmark: full Loader iteration -- DIRECT (C+mmap) vs ZARR.

Creates temporary zarr stores (dense + sparse) and times complete iteration
through the Loader at various chunk_size values.

Run with:
    python tests/bench_loader.py
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
                N_OBS, N_VARS, density=0.1, format="csr", dtype="f4", random_state=np.random.default_rng(0)
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


def bench_loader(store_path: Path, chunk_size: int, preload_nchunks: int, batch_size: int, sparse: bool, force_zarr: bool = False) -> dict:
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
    )
    loader.add_anndata(adata)

    if force_zarr:
        loader._can_direct_read = staticmethod(lambda _: False)

    n_batches = 0
    total_obs = 0
    t0 = time.perf_counter()
    for batch in loader:
        n_batches += 1
        total_obs += batch["X"].shape[0]
    elapsed = time.perf_counter() - t0
    return {"elapsed": elapsed, "n_batches": n_batches, "total_obs": total_obs}


def run_suite(store_path: Path, label: str, sparse: bool, configs: list[dict]):
    for mode, force_zarr in [("DIRECT (C+mmap)", False), ("ZARR (async)", True)]:
        print(f"\n{'=' * 85}")
        print(f"  {label} [{mode}]")
        print(f"  {N_OBS} obs x {N_VARS} vars, repeats={REPEATS}, C ext={_HAS_C_EXT}")
        print(f"{'=' * 85}")
        print(
            f"{'chunk_size':>10}  {'preload':>7}  {'batch':>5}  {'n_batches':>9}  "
            f"{'best (s)':>8}  {'mean (s)':>8}  {'obs/s':>10}"
        )
        print("-" * 85)

        for cfg in configs:
            times = []
            result = None
            for _ in range(REPEATS):
                result = bench_loader(store_path, sparse=sparse, force_zarr=force_zarr, **cfg)
                times.append(result["elapsed"])

            best = min(times)
            mean = sum(times) / len(times)
            obs_per_sec = result["total_obs"] / best

            print(
                f"{cfg['chunk_size']:>10}  {cfg['preload_nchunks']:>7}  "
                f"{cfg['batch_size']:>5}  {result['n_batches']:>9}  "
                f"{best:>8.4f}  {mean:>8.4f}  {obs_per_sec:>10.0f}"
            )


def main():
    configs = [
        {"chunk_size": 2, "preload_nchunks": 512, "batch_size": 2},
        {"chunk_size": 4, "preload_nchunks": 256, "batch_size": 4},
        {"chunk_size": 8, "preload_nchunks": 128, "batch_size": 8},
        {"chunk_size": 16, "preload_nchunks": 64, "batch_size": 16},
        {"chunk_size": 32, "preload_nchunks": 32, "batch_size": 32},
    ]

    with tempfile.TemporaryDirectory() as tmp:
        store_path = _create_store(Path(tmp))
        run_suite(store_path, "Dense (f32)", sparse=False, configs=configs)
        run_suite(store_path, "Sparse CSR (f32, density=0.1)", sparse=True, configs=configs)


if __name__ == "__main__":
    main()

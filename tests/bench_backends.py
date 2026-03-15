"""Head-to-head benchmark: zarrs (Rust) mmap vs pread vs C ext vs zarr-python async.

Sweeps over dataset size, feature width (shard file size), and access pattern
to find where mmap vs pread diverges.

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
import zarr

from annbatch import ChunkSampler, Loader, write_sharded
from annbatch._direct_read import _HAS_C_EXT

REPEATS = 3


def _create_store(
    tmp: Path,
    n_obs: int,
    n_vars: int,
    dense_chunk: int,
    dense_shard: int,
) -> Path:
    store_path = tmp / f"bench_{n_obs}x{n_vars}.zarr"
    rng = np.random.default_rng(0)
    adata = ad.AnnData(
        X=rng.standard_normal((n_obs, n_vars)).astype("f4"),
        obs=pd.DataFrame(
            {"label": rng.integers(0, 5, size=n_obs)},
            index=[str(i) for i in range(n_obs)],
        ),
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_vars)]),
    )
    g = zarr.open_group(store_path, mode="w", zarr_format=3)
    write_sharded(
        g, adata,
        dense_chunk_size=dense_chunk,
        dense_shard_size=dense_shard,
    )
    return store_path


def _open_dense(path: Path) -> ad.AnnData:
    g = zarr.open(path)
    return ad.AnnData(
        X=g["X"],
        obs=ad.io.read_elem(g["obs"]),
        var=ad.io.read_elem(g["var"]),
    )


BACKENDS = [
    ("zarrs mmap+fuse", "zarrs", True, True),
    ("zarrs pread+fuse", "zarrs", True, False),
    ("zarrs mmap nofuse", "zarrs", False, True),
    ("zarrs pread nofuse", "zarrs", False, False),
    ("C ext (mmap)", "c", True, True),
    ("zarr-python async", None, True, True),
]


def bench_loader(store_path, chunk_size, preload_nchunks, batch_size,
                  sync_backend, fuse_ranges=True, use_mmap=True):
    if sync_backend == "zarrs":
        import zarr_direct.reader as _zdr
        _zdr._reader_cache.clear()
        _zdr._FUSE_DEFAULT = fuse_ranges
        _zdr._MMAP_DEFAULT = use_mmap

    adata = _open_dense(store_path)
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


def run_suite(store_path, label, n_obs, n_vars, loader_cfg):
    cs = loader_cfg["chunk_size"]
    pn = loader_cfg["preload_nchunks"]
    bs = loader_cfg["batch_size"]

    shard_bytes = n_vars * 4 * loader_cfg.get("dense_shard", 128)
    print(f"\n{'=' * 120}")
    print(f"  {label}  --  {n_obs} obs x {n_vars} vars  "
          f"(~{shard_bytes / 1024:.0f} KB/shard)  "
          f"chunk={cs} preload={pn} batch={bs}  repeats={REPEATS}")
    print(f"{'=' * 120}")
    print(f"  {'Backend':<25}  {'best (s)':>10}  {'mean (s)':>10}  "
          f"{'obs/s':>12}  {'vs async':>10}  {'vs C ext':>10}  {'mmap/pread':>10}")
    print(f"  {'-' * 110}")

    results = {}
    for name, sb, fuse, mmap in BACKENDS:
        if sb == "c" and not _HAS_C_EXT:
            results[name] = None
            continue
        times = []
        res = None
        for _ in range(REPEATS):
            res = bench_loader(store_path, sync_backend=sb,
                                fuse_ranges=fuse, use_mmap=mmap,
                                chunk_size=cs, preload_nchunks=pn, batch_size=bs)
            times.append(res["elapsed"])
        best = min(times)
        mean = sum(times) / len(times)
        obs_per_sec = res["total_obs"] / best
        results[name] = (best, mean, obs_per_sec, res["total_obs"])

    async_best = results.get("zarr-python async")
    c_best = results.get("C ext (mmap)")
    mmap_fuse = results.get("zarrs mmap+fuse")
    pread_fuse = results.get("zarrs pread+fuse")

    for name, _, _, _ in BACKENDS:
        vals = results.get(name)
        if vals is None:
            print(f"  {name:<25}  {'N/A':>10}  {'N/A':>10}  {'N/A':>12}  {'N/A':>10}  {'N/A':>10}  {'':>10}")
            continue
        best, mean, obs_per_sec, _ = vals
        vs_async = f"{async_best[0] / best:.2f}x" if async_best else "N/A"
        vs_c = f"{c_best[0] / best:.2f}x" if c_best else "N/A"
        if mmap_fuse and pread_fuse and "mmap" in name and "fuse" in name and "nofuse" not in name:
            ratio = f"{pread_fuse[0] / mmap_fuse[0]:.2f}x"
        elif mmap_fuse and pread_fuse and "pread" in name and "fuse" in name and "nofuse" not in name:
            ratio = f"{mmap_fuse[0] / pread_fuse[0]:.2f}x"
        else:
            ratio = ""
        print(f"  {name:<25}  {best:>10.4f}  {mean:>10.4f}  {obs_per_sec:>12.0f}  {vs_async:>10}  {vs_c:>10}  {ratio:>10}")

    print()


def main():
    scenarios = [
        {
            "label": "Small narrow (10k x 200)",
            "n_obs": 10_000, "n_vars": 200,
            "dense_chunk": 64, "dense_shard": 128,
            "loader": {"chunk_size": 2, "preload_nchunks": 512, "batch_size": 32},
        },
        {
            "label": "Small wide (10k x 2000)",
            "n_obs": 10_000, "n_vars": 2_000,
            "dense_chunk": 64, "dense_shard": 128,
            "loader": {"chunk_size": 2, "preload_nchunks": 512, "batch_size": 32},
        },
        {
            "label": "Medium wide (50k x 2000)",
            "n_obs": 50_000, "n_vars": 2_000,
            "dense_chunk": 64, "dense_shard": 256,
            "loader": {"chunk_size": 2, "preload_nchunks": 512, "batch_size": 32},
        },
        {
            "label": "Large wide (50k x 5000)",
            "n_obs": 50_000, "n_vars": 5_000,
            "dense_chunk": 64, "dense_shard": 512,
            "loader": {"chunk_size": 2, "preload_nchunks": 256, "batch_size": 32},
        },
        {
            "label": "Small narrow, tiny batch (10k x 200)",
            "n_obs": 10_000, "n_vars": 200,
            "dense_chunk": 64, "dense_shard": 128,
            "loader": {"chunk_size": 1, "preload_nchunks": 1024, "batch_size": 4},
        },
    ]

    with tempfile.TemporaryDirectory() as tmp:
        for sc in scenarios:
            store_path = _create_store(
                Path(tmp), sc["n_obs"], sc["n_vars"],
                sc["dense_chunk"], sc["dense_shard"],
            )
            lc = sc["loader"].copy()
            lc["dense_shard"] = sc["dense_shard"]
            run_suite(store_path, sc["label"], sc["n_obs"], sc["n_vars"], lc)

    print("Done.")


if __name__ == "__main__":
    main()

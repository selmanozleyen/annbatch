"""Benchmark: mmap vs pread vs C ext vs zarr-python async.

Two modes:
  1. Micro-bench: direct reader calls with controlled shard sizes and access patterns
  2. Loader-bench (optional, --loader): end-to-end through annbatch Loader

Run with:
    python tests/bench_backends.py            # micro-bench only (fast)
    python tests/bench_backends.py --loader   # also run loader end-to-end
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import zarr
from zarr.codecs import BloscCodec

REPEATS = 3
N_SAMPLE_ITERS = 10


# ---------------------------------------------------------------------------
# Store creation
# ---------------------------------------------------------------------------

def _create_zarr(
    path: Path, n_obs: int, n_vars: int,
    inner_chunk: int, shard_size: int,
) -> zarr.Array:
    store = zarr.storage.LocalStore(path)
    g = zarr.open_group(store, mode="w", zarr_format=3)
    arr = g.create_array(
        "X",
        shape=(n_obs, n_vars),
        dtype="float32",
        chunks=(inner_chunk, n_vars),
        shards=(shard_size, n_vars),
        compressors=(BloscCodec(cname="lz4", clevel=3),),
    )
    rng = np.random.default_rng(42)
    for start in range(0, n_obs, shard_size):
        end = min(start + shard_size, n_obs)
        arr[start:end] = rng.standard_normal((end - start, n_vars)).astype("f4")
    return zarr.open(store)["X"]


# ---------------------------------------------------------------------------
# Generate access patterns
# ---------------------------------------------------------------------------

def make_random_ranges(n_obs, chunk_size, n_ranges, rng):
    max_start = n_obs - chunk_size
    starts = rng.integers(0, max_start, size=n_ranges).astype(np.int64)
    stops = (starts + chunk_size).astype(np.int64)
    return starts, stops


def make_sequential_ranges(n_obs, chunk_size, n_ranges):
    starts = np.arange(0, min(n_ranges * chunk_size, n_obs), chunk_size, dtype=np.int64)[:n_ranges]
    stops = (starts + chunk_size).astype(np.int64)
    np.clip(stops, 0, n_obs, out=stops)
    return starts, stops


# ---------------------------------------------------------------------------
# Micro-benchmark runners
# ---------------------------------------------------------------------------

def bench_zarrs_direct(arr, all_starts, all_stops, use_mmap, fuse_ranges):
    from zarr_direct.reader import _parse_store_path, ShardedArrayReader
    store_root, array_path = _parse_store_path(arr)
    reader = ShardedArrayReader(store_root, use_mmap=use_mmap, fuse_ranges=fuse_ranges)

    times = []
    for starts, stops in zip(all_starts, all_stops):
        s = np.ascontiguousarray(starts, dtype=np.int64)
        e = np.ascontiguousarray(stops, dtype=np.int64)
        t0 = time.perf_counter()
        reader.read_raw(array_path, s, e)
        times.append(time.perf_counter() - t0)
    return times


def bench_c_ext(arr, all_starts, all_stops):
    from annbatch._direct_read import read_direct_dense, _HAS_C_EXT
    if not _HAS_C_EXT:
        return None
    times = []
    for starts, stops in zip(all_starts, all_stops):
        boundaries = np.empty(len(starts) * 2, dtype=np.int64)
        boundaries[0::2] = starts
        boundaries[1::2] = stops
        t0 = time.perf_counter()
        read_direct_dense(arr, boundaries)
        times.append(time.perf_counter() - t0)
    return times


def bench_zarr_async(arr, all_starts, all_stops):
    times = []
    for starts, stops in zip(all_starts, all_stops):
        t0 = time.perf_counter()
        chunks = [arr[int(s):int(e)] for s, e in zip(starts, stops)]
        _ = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        times.append(time.perf_counter() - t0)
    return times


# ---------------------------------------------------------------------------
# Micro-benchmark suite
# ---------------------------------------------------------------------------

def run_micro(arr, label, n_obs, n_vars, shard_size, inner_chunk,
              chunk_size, n_ranges, pattern):
    rng = np.random.default_rng(0)
    all_starts, all_stops = [], []
    for _ in range(N_SAMPLE_ITERS):
        if pattern == "random":
            s, e = make_random_ranges(n_obs, chunk_size, n_ranges, rng)
        else:
            s, e = make_sequential_ranges(n_obs, chunk_size, n_ranges)
        all_starts.append(s)
        all_stops.append(e)

    shard_mb = shard_size * n_vars * 4 / 1024 / 1024
    print(f"\n{'=' * 115}")
    print(f"  {label}")
    print(f"  {n_obs} obs x {n_vars} vars  |  shard={shard_size} rows ({shard_mb:.1f} MB)  "
          f"|  inner_chunk={inner_chunk}  |  {pattern} {n_ranges} x {chunk_size}-row ranges  "
          f"|  {N_SAMPLE_ITERS} iters x {REPEATS} repeats")
    print(f"{'=' * 115}")

    backends = [
        ("zarrs mmap+fuse",    lambda: bench_zarrs_direct(arr, all_starts, all_stops, True, True)),
        ("zarrs pread+fuse",   lambda: bench_zarrs_direct(arr, all_starts, all_stops, False, True)),
        ("zarrs mmap nofuse",  lambda: bench_zarrs_direct(arr, all_starts, all_stops, True, False)),
        ("zarrs pread nofuse", lambda: bench_zarrs_direct(arr, all_starts, all_stops, False, False)),
        ("C ext (mmap)",       lambda: bench_c_ext(arr, all_starts, all_stops)),
        ("zarr-python async",  lambda: bench_zarr_async(arr, all_starts, all_stops)),
    ]

    results = {}
    for name, fn in backends:
        best_totals = []
        for _ in range(REPEATS):
            t = fn()
            if t is None:
                break
            best_totals.append(sum(t))
        if best_totals:
            best = min(best_totals)
            mean = sum(best_totals) / len(best_totals)
            results[name] = (best, mean)
        else:
            results[name] = None

    async_r = results.get("zarr-python async")
    c_r = results.get("C ext (mmap)")
    mmap_r = results.get("zarrs mmap+fuse")
    pread_r = results.get("zarrs pread+fuse")

    print(f"  {'Backend':<25}  {'total (s)':>10}  {'mean (s)':>10}  "
          f"{'vs async':>10}  {'vs C ext':>10}  {'mmap/pread':>12}")
    print(f"  {'-' * 105}")

    for name, _ in backends:
        r = results.get(name)
        if r is None:
            print(f"  {name:<25}  {'N/A':>10}  {'N/A':>10}  {'N/A':>10}  {'N/A':>10}  {'':>12}")
            continue
        best, mean = r
        vs_a = f"{async_r[0] / best:.1f}x" if async_r else "N/A"
        vs_c = f"{c_r[0] / best:.1f}x" if c_r else "N/A"
        ratio = ""
        if mmap_r and pread_r:
            if "mmap" in name and "fuse" in name and "nofuse" not in name:
                ratio = f"mmap {pread_r[0] / mmap_r[0]:.2f}x"
            elif "pread" in name and "fuse" in name and "nofuse" not in name:
                ratio = f"pread {mmap_r[0] / pread_r[0]:.2f}x"
        print(f"  {name:<25}  {best:>10.4f}  {mean:>10.4f}  {vs_a:>10}  {vs_c:>10}  {ratio:>12}")


# ---------------------------------------------------------------------------
# Loader benchmark (optional)
# ---------------------------------------------------------------------------

def run_loader_suite(tmp):
    import anndata as ad
    import pandas as pd
    from annbatch import ChunkSampler, Loader, write_sharded
    from annbatch._direct_read import _HAS_C_EXT

    n_obs, n_vars = 10_000, 200
    store_path = tmp / "loader.zarr"
    rng = np.random.default_rng(0)
    adata = ad.AnnData(
        X=rng.standard_normal((n_obs, n_vars)).astype("f4"),
        obs=pd.DataFrame({"label": rng.integers(0, 5, size=n_obs)},
                          index=[str(i) for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_vars)]),
    )
    g = zarr.open_group(store_path, mode="w", zarr_format=3)
    write_sharded(g, adata, dense_chunk_size=64, dense_shard_size=128)

    backends = [
        ("zarrs mmap+fuse", "zarrs", True, True),
        ("zarrs pread+fuse", "zarrs", True, False),
        ("C ext (mmap)", "c", True, True),
        ("zarr-python async", None, True, True),
    ]

    print(f"\n{'=' * 115}")
    print(f"  Loader end-to-end  --  {n_obs} x {n_vars}  chunk=2  preload=512  batch=32")
    print(f"{'=' * 115}")
    print(f"  {'Backend':<25}  {'best (s)':>10}  {'mean (s)':>10}")
    print(f"  {'-' * 50}")

    for name, sb, fuse, mmap in backends:
        if sb == "c" and not _HAS_C_EXT:
            print(f"  {name:<25}  {'N/A':>10}")
            continue
        if sb == "zarrs":
            import zarr_direct.reader as _zdr
            _zdr._reader_cache.clear()
            _zdr._FUSE_DEFAULT = fuse
            _zdr._MMAP_DEFAULT = mmap

        times = []
        for _ in range(3):
            adata_open = ad.AnnData(
                X=zarr.open(store_path)["X"],
                obs=ad.io.read_elem(zarr.open(store_path)["obs"]),
                var=ad.io.read_elem(zarr.open(store_path)["var"]),
            )
            sampler = ChunkSampler(chunk_size=2, preload_nchunks=512, batch_size=32, shuffle=False)
            loader = Loader(batch_sampler=sampler, return_index=True,
                            preload_to_gpu=False, to_torch=False, sync_backend=sb)
            loader.add_anndata(adata_open)
            t0 = time.perf_counter()
            for batch in loader:
                pass
            times.append(time.perf_counter() - t0)

        print(f"  {name:<25}  {min(times):>10.4f}  {sum(times)/len(times):>10.4f}")
    print()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    scenarios = [
        {
            "label": "Tiny shard (~100 KB), random",
            "n_obs": 10_000, "n_vars": 200,
            "inner_chunk": 64, "shard_size": 128,
            "chunk_size": 2, "n_ranges": 256, "pattern": "random",
        },
        {
            "label": "Medium shard (~1 MB), random",
            "n_obs": 10_000, "n_vars": 2_000,
            "inner_chunk": 64, "shard_size": 128,
            "chunk_size": 2, "n_ranges": 128, "pattern": "random",
        },
        {
            "label": "Big shard (~10 MB), random",
            "n_obs": 10_000, "n_vars": 5_000,
            "inner_chunk": 64, "shard_size": 512,
            "chunk_size": 2, "n_ranges": 128, "pattern": "random",
        },
        {
            "label": "Big shard (~10 MB), many tiny ranges",
            "n_obs": 10_000, "n_vars": 5_000,
            "inner_chunk": 64, "shard_size": 512,
            "chunk_size": 1, "n_ranges": 512, "pattern": "random",
        },
    ]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for sc in scenarios:
            path = tmp / f"z_{sc['n_obs']}_{sc['n_vars']}_{sc['shard_size']}.zarr"
            arr = _create_zarr(
                path, sc["n_obs"], sc["n_vars"],
                sc["inner_chunk"], sc["shard_size"],
            )
            run_micro(
                arr, sc["label"], sc["n_obs"], sc["n_vars"],
                sc["shard_size"], sc["inner_chunk"],
                sc["chunk_size"], sc["n_ranges"], sc["pattern"],
            )

        if "--loader" in sys.argv:
            run_loader_suite(tmp)

    print("Done.")


if __name__ == "__main__":
    main()

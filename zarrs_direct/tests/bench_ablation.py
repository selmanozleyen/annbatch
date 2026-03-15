"""Ablation benchmark: measure contribution of each speedup layer.

Configurations:
  1. zarr-python async   -- baseline, uses zarr-python's internal async path
  2. zarrs + pread       -- zarrs via PyO3, FilesystemStore (pread, no caching)
  3. zarrs + mmap        -- zarrs via PyO3, MmapStore (zero-copy, cached)
  4. annbatch C ext      -- reference, the existing C extension direct reader

Each config reads the same sharded array with the same row ranges.

Run with:
    python tests/bench_ablation.py

Requires zarr-direct to be installed (maturin develop).
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
import zarr

N_OBS = 10_000
N_VARS = 200
REPEATS = 3


def _create_store(path: Path) -> Path:
    """Create a sharded zarr store for benchmarking."""
    from zarr.codecs import BloscCodec

    store = zarr.storage.LocalStore(path)
    g = zarr.open_group(store, mode="w", zarr_format=3)

    g.create_array(
        "X",
        shape=(N_OBS, N_VARS),
        dtype="float32",
        chunks=(64, N_VARS),
        shards=(128, N_VARS),
        compressors=(BloscCodec(cname="lz4", clevel=3),),
    )

    rng = np.random.default_rng(42)
    arr = zarr.open(store)["X"]
    arr[:] = rng.standard_normal((N_OBS, N_VARS)).astype("float32")

    return path


def _make_boundaries(chunk_size: int, preload_nchunks: int) -> np.ndarray:
    """Build interleaved [s0, e0, s1, e1, ...] boundaries."""
    total = chunk_size * preload_nchunks
    starts = np.arange(0, min(total, N_OBS), chunk_size)
    stops = np.minimum(starts + chunk_size, N_OBS)
    boundaries = np.empty(len(starts) * 2, dtype=np.int64)
    boundaries[0::2] = starts
    boundaries[1::2] = stops
    return boundaries


def bench_zarr_python_async(arr: zarr.Array, boundaries: np.ndarray) -> float:
    """Read via zarr-python's standard path (includes async overhead)."""
    starts = boundaries[0::2]
    stops = boundaries[1::2]
    t0 = time.perf_counter()
    chunks = []
    for s, e in zip(starts, stops):
        chunks.append(arr[s:e])
    _ = np.concatenate(chunks)
    return time.perf_counter() - t0


def bench_zarrs_pread(arr: zarr.Array, boundaries: np.ndarray) -> float:
    """Read via zarrs + FilesystemStore (pread)."""
    from zarr_direct import read_direct_dense

    t0 = time.perf_counter()
    _ = read_direct_dense(arr, boundaries, use_mmap=False)
    return time.perf_counter() - t0


def bench_zarrs_mmap(arr: zarr.Array, boundaries: np.ndarray) -> float:
    """Read via zarrs + MmapStore (zero-copy mmap)."""
    from zarr_direct import read_direct_dense

    t0 = time.perf_counter()
    _ = read_direct_dense(arr, boundaries, use_mmap=True)
    return time.perf_counter() - t0


def bench_annbatch_c_ext(arr: zarr.Array, boundaries: np.ndarray) -> float | None:
    """Read via annbatch's C extension (reference)."""
    try:
        from annbatch._direct_read import read_direct_dense, _HAS_C_EXT

        if not _HAS_C_EXT:
            return None
    except ImportError:
        return None

    t0 = time.perf_counter()
    _ = read_direct_dense(arr, boundaries)
    return time.perf_counter() - t0


def run_config(arr: zarr.Array, chunk_size: int, preload_nchunks: int, batch_size: int):
    """Run all backends for one configuration."""
    boundaries = _make_boundaries(chunk_size, preload_nchunks)
    total_obs = (boundaries[1::2] - boundaries[0::2]).sum()

    benches = {
        "zarr-python async": bench_zarr_python_async,
        "zarrs + pread":     bench_zarrs_pread,
        "zarrs + mmap":      bench_zarrs_mmap,
        "annbatch C ext":    bench_annbatch_c_ext,
    }

    results = {}
    for name, fn in benches.items():
        times = []
        for _ in range(REPEATS):
            t = fn(arr, boundaries)
            if t is None:
                break
            times.append(t)
        if times:
            best = min(times)
            mean = sum(times) / len(times)
            obs_per_sec = total_obs / best
            results[name] = (best, mean, obs_per_sec)
        else:
            results[name] = None

    return results, total_obs


def main():
    configs = [
        {"chunk_size": 2,  "preload_nchunks": 512, "batch_size": 2},
        {"chunk_size": 4,  "preload_nchunks": 256, "batch_size": 4},
        {"chunk_size": 8,  "preload_nchunks": 128, "batch_size": 8},
        {"chunk_size": 16, "preload_nchunks": 64,  "batch_size": 16},
        {"chunk_size": 32, "preload_nchunks": 32,  "batch_size": 32},
    ]

    with tempfile.TemporaryDirectory() as tmp:
        store_path = _create_store(Path(tmp))
        store = zarr.storage.LocalStore(store_path)
        arr = zarr.open(store)["X"]

        print(f"\nAblation Benchmark: {N_OBS} obs x {N_VARS} vars, repeats={REPEATS}")
        print(f"{'=' * 100}")

        for cfg in configs:
            print(
                f"\nchunk_size={cfg['chunk_size']:>3}  "
                f"preload_nchunks={cfg['preload_nchunks']:>3}  "
                f"batch_size={cfg['batch_size']:>3}"
            )
            print(f"{'-' * 100}")
            print(
                f"  {'Backend':<25}  {'best (s)':>10}  {'mean (s)':>10}  "
                f"{'obs/s':>12}  {'speedup':>8}"
            )
            print(f"  {'-' * 90}")

            results, total_obs = run_config(arr, **cfg)

            baseline = results.get("zarr-python async")
            baseline_best = baseline[0] if baseline else 1.0

            for name, vals in results.items():
                if vals is None:
                    print(f"  {name:<25}  {'N/A':>10}  {'N/A':>10}  {'N/A':>12}  {'N/A':>8}")
                else:
                    best, mean, obs_per_sec = vals
                    speedup = baseline_best / best if baseline else 0.0
                    print(
                        f"  {name:<25}  {best:>10.4f}  {mean:>10.4f}  "
                        f"{obs_per_sec:>12.0f}  {speedup:>7.2f}x"
                    )

        print(f"\n{'=' * 100}")
        print("Done.")


if __name__ == "__main__":
    main()

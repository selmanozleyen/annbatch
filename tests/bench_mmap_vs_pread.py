"""Quick benchmark: mmap vs pread for the C shard reader.

Usage:
    python tests/bench_mmap_vs_pread.py
    python tests/bench_mmap_vs_pread.py --n_obs 200000 --n_vars 2000
"""
from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import zarr
from zarr.codecs import BloscCodec


def create_test_array(
    path: Path, n_obs: int, n_vars: int,
    inner_chunk: int = 512, shard_size: int = 8192,
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


def make_boundaries(starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    b = np.empty(len(starts) * 2, dtype=np.int64)
    b[0::2] = starts
    b[1::2] = stops
    return b


def bench(fn, arr, all_boundaries, warmup=1, repeats=5):
    for _ in range(warmup):
        fn(arr, all_boundaries[0])

    times = []
    for boundaries in all_boundaries:
        t0 = time.perf_counter()
        out = fn(arr, boundaries)
        times.append(time.perf_counter() - t0)
    return times, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_obs", type=int, default=100_000)
    parser.add_argument("--n_vars", type=int, default=1000)
    parser.add_argument("--inner_chunk", type=int, default=512)
    parser.add_argument("--shard_size", type=int, default=8192)
    parser.add_argument("--batch_rows", type=int, default=4096)
    parser.add_argument("--n_iters", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--pattern", choices=["random", "sequential"], default="random")
    args = parser.parse_args()

    from annbatch._direct_read import (
        read_direct_dense, read_pread_dense, _HAS_C_EXT,
    )

    print(f"C extension: {_HAS_C_EXT}")
    print(f"Array: {args.n_obs} x {args.n_vars}, shard={args.shard_size}, "
          f"inner_chunk={args.inner_chunk}")
    print(f"Batch: {args.batch_rows} rows, {args.n_iters} iterations, "
          f"pattern={args.pattern}")
    print()

    with tempfile.TemporaryDirectory() as tmp:
        print("Creating test array...", end=" ", flush=True)
        arr = create_test_array(
            Path(tmp) / "test.zarr", args.n_obs, args.n_vars,
            args.inner_chunk, args.shard_size,
        )
        print("done.")

        rng = np.random.default_rng(0)
        all_boundaries = []
        for _ in range(args.n_iters):
            if args.pattern == "random":
                starts = rng.integers(0, args.n_obs - args.batch_rows,
                                      size=1).astype(np.int64)
            else:
                starts = np.array([0], dtype=np.int64)
            stops = (starts + args.batch_rows).astype(np.int64)
            np.clip(stops, 0, args.n_obs, out=stops)
            all_boundaries.append(make_boundaries(starts, stops))

        print(f"\n{'Backend':<20} {'Mean (ms)':>10} {'Std (ms)':>10} {'Min (ms)':>10}")
        print("-" * 55)

        for name, fn in [("mmap", read_direct_dense), ("pread", read_pread_dense)]:
            times, out = bench(fn, arr, all_boundaries, warmup=1, repeats=args.repeats)
            ms = np.array(times) * 1000
            print(f"{name:<20} {ms.mean():>10.2f} {ms.std():>10.2f} {ms.min():>10.2f}")

        # Correctness check
        ref = read_direct_dense(arr, all_boundaries[0])
        check = read_pread_dense(arr, all_boundaries[0])
        if np.array_equal(ref, check):
            print("\nCorrectness: PASS (mmap == pread)")
        else:
            print("\nCorrectness: FAIL!")
            print(f"  Max abs diff: {np.abs(ref - check).max()}")


if __name__ == "__main__":
    main()
